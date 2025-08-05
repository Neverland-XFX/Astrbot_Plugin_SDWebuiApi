# astrbot_plugin_sdwebuiapi/main.py
from __future__ import annotations
import base64
import datetime as dt
import os, uuid, re, asyncio
from pathlib import Path
from typing import List, Optional, Dict, Any

import httpx
from colorama import Fore, Style, init as colorama_init

from astrbot.api.star import Star, register, Context
from astrbot.api.event import filter, AstrMessageEvent
import astrbot.api.message_components as Comp
from astrbot.api import logger

# ─────────────────────────────── 工具：本地图片 → Comp.Image ────────────────────────────────
def build_image_component(filepath: Path) -> Comp.MessageSegment:
    path_str = str(filepath.resolve())
    if hasattr(Comp.Image, "fromPath"):
        return Comp.Image.fromPath(path_str)
    if hasattr(Comp.Image, "fromFile"):
        return Comp.Image.fromFile(path_str)
    if hasattr(Comp.Image, "fromFilePath"):
        return Comp.Image.fromFilePath(path_str)
    # fallback
    try:
        b64 = base64.b64encode(filepath.read_bytes()).decode()
        if hasattr(Comp.Image, "fromBase64"):
            return Comp.Image.fromBase64(b64)
        if hasattr(Comp.Image, "fromBytes"):
            return Comp.Image.fromBytes(base64.b64decode(b64), f"{uuid.uuid4()}.png")
    except Exception as e:
        logger.warning(f"[sdWebui] Base64 兜底失败：{e}")
    return Comp.Plain(f"[图片保存在本地 {filepath}] (当前平台无本地图片发送能力)")

# ─────────────────────────────── 插件实现 ────────────────────────────────
@register(
    "sdWebuiApi",
    "neverland",
    "Stable Diffusion WebUI 全功能插件",
    "1.0.0"
)
class SdWebuiApi(Star):
    def __init__(self, context: Context, config: dict, *_):
        colorama_init()
        self.ctx: Context = context
        self.config = config
        # 读取配置
        self.base_url = config.get("base_url", "http://127.0.0.1:7860").rstrip("/")
        self.img_dir  = (Path(__file__).parent / config.get("output_dir", "images")).resolve()
        self.img_dir.mkdir(parents=True, exist_ok=True)
        self.steps    = int(config.get("steps", 20))
        self.cfg      = float(config.get("cfg", 7.0))
        self.w        = int(config.get("width", 512))
        self.h        = int(config.get("height", 512))
        self.sampler  = str(config.get("sampler", "Euler a"))
        self.batch    = int(config.get("batch", 1))
        self.n_iter   = int(config.get("n_iter", 1))
        self.upscaler = config.get("upscaler", "")
        self.upscale_factor = float(config.get("upscale_factor", 2.0))
        self.hires_fix = bool(config.get("hires_fix", False))
        self.max_tasks = int(config.get("max_concurrent_tasks", 3))
        self.timeout = int(config.get("session_timeout", 180))
        self.negative_prompt = config.get("negative_prompt_global", "")
        self.positive_prompt = config.get("positive_prompt_global", "")
        self.show_positive = bool(config.get("show_positive_prompt", False))
        self.auto_tr  = bool(config.get("translate_prompt", False))
        self.llm_gen  = bool(config.get("llm_generate_prompt", False))
        self.verbose  = bool(config.get("verbose", True))
        # 并发任务队列
        self.task_semaphore = asyncio.Semaphore(self.max_tasks)
        # 缓存
        self._models = []
        self._samplers = []
        self._upscalers = []
        self._lora = []
        self._embedding = []
        self._lora_weights: Dict[str, float] = {}
        super().__init__(context)

    # ────────────── WebUI 资源查询（模型、采样器、upscaler、lora、embedding） ──────────────
    async def fetch_webui_resource(self, key: str) -> List[str]:
        table = {
            "model": "/sdapi/v1/sd-models",
            "sampler": "/sdapi/v1/samplers",
            "upscaler": "/sdapi/v1/upscalers",
            "lora": "/sdapi/v1/loras",
            "embedding": "/sdapi/v1/embeddings"
        }
        url = self.base_url + table[key]
        async with httpx.AsyncClient(timeout=self.timeout) as cli:
            resp = await cli.get(url)
            resp.raise_for_status()
            result = await resp.aread()
            js = resp.json()
            if key == "model":
                return [m["model_name"] for m in js]
            if key == "sampler":
                return [m["name"] for m in js]
            if key == "upscaler":
                return [m["name"] for m in js]
            if key == "lora":
                return [m["name"] for m in js]
            if key == "embedding":
                return list(js.get("loaded", {}).keys())
        return []

    # ────────────── 辅助 — prompt 翻译 / LLM自动生图提示词 ──────────────
    async def translate_prompt(self, text: str) -> str:
        if not self.auto_tr or not any("\u4e00" <= ch <= "\u9fff" for ch in text):
            return text
        rsp = await self.ctx.get_using_provider().text_chat(prompt=text)
        translated = rsp.result_chain.chain[0].text.strip()
        logger.info(f"[sdWebui] 翻译：{text} → {translated}")
        return translated or text

    async def llm_generate_prompt(self, user_text: str) -> str:
        if not self.llm_gen:
            return user_text
        prompt_hint = (
            "请根据以下描述生成用于 Stable Diffusion WebUI 的英文提示词，只输出英文逗号分隔的 prompt 字符串，无需解释性文本：\n"
        )
        content = prompt_hint + user_text
        rsp = await self.ctx.get_using_provider().text_chat(prompt=content)
        out = rsp.result_chain.chain[0].text.strip()
        logger.info(f"[sdWebui] LLM生成：{user_text} → {out}")
        return out or user_text

    # ────────────── 负向提示词、拼接全局正向提示词 ──────────────
    def merge_prompt(self, user_prompt: str) -> Dict[str, str]:
        user_prompt = user_prompt.strip()
        if self.auto_tr:
            # 会在命令逻辑里自动翻译或 LLM
            pass
        positive = self.positive_prompt.strip() + " " + user_prompt
        negative = self.negative_prompt
        return dict(positive=positive.strip(), negative=negative.strip())

    # ────────────── txt2img ──────────────
    async def txt2img(
        self, prompt: str, negative: str = "", n_iter: int = None, batch_size: int = None,
        width: int = None, height: int = None, steps: int = None, cfg_scale: float = None,
        sampler: str = None, hires_fix: bool = None, lora: List[str] = None, lora_weights: Dict[str, float] = None
    ) -> List[Path]:
        payload = {
            "prompt": prompt,
            "negative_prompt": negative,
            "steps": steps or self.steps,
            "cfg_scale": cfg_scale or self.cfg,
            "sampler_index": sampler or self.sampler,
            "width": width or self.w,
            "height": height or self.h,
            "batch_size": batch_size or self.batch,
            "n_iter": n_iter or self.n_iter,
            "seed": -1
        }
        # hires fix
        if hires_fix if hires_fix is not None else self.hires_fix:
            payload["enable_hr"] = True
            payload["hr_scale"] = 2.0
            payload["hr_upscaler"] = self.upscaler or "Latent"
            payload["hr_second_pass_steps"] = int(payload["steps"] / 2)
        # LoRA 支持
        if lora and lora_weights:
            payload.setdefault("alwayson_scripts", {})["additional_networks"] = {
                "args": [
                    {
                        "lora": ln,
                        "weight": lora_weights.get(ln, 1.0)
                    }
                    for ln in lora
                ]
            }
        url = f"{self.base_url}/sdapi/v1/txt2img"
        logger.info(f"[sdWebui] POST {url}  payload={payload}")
        async with httpx.AsyncClient(timeout=self.timeout) as cli:
            resp = await cli.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        images = data.get("images", [])
        if not images:
            raise RuntimeError("WebUI 未返回 images")
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        paths: List[Path] = []
        for idx, b64 in enumerate(images, start=1):
            if "," in b64:
                b64 = b64.split(",", 1)[1]
            img_bytes = base64.b64decode(b64)
            file_path = self.img_dir / f"{ts}_{idx}.png"
            file_path.write_bytes(img_bytes)
            paths.append(file_path)
        logger.info(f"[sdWebui] 保存 {len(paths)} 张图片 → {self.img_dir}")
        return paths

    # ────────────── img2img ──────────────
    async def img2img(self, image_bytes: bytes, prompt: str, negative: str = "", **kwargs) -> List[Path]:
        payload = {
            "init_images": [base64.b64encode(image_bytes).decode()],
            "prompt": prompt,
            "negative_prompt": negative,
            "steps": kwargs.get("steps", self.steps),
            "cfg_scale": kwargs.get("cfg_scale", self.cfg),
            "sampler_index": kwargs.get("sampler", self.sampler),
            "width": kwargs.get("width", self.w),
            "height": kwargs.get("height", self.h),
            "batch_size": kwargs.get("batch_size", self.batch),
            "n_iter": kwargs.get("n_iter", self.n_iter),
            "seed": -1
        }
        url = f"{self.base_url}/sdapi/v1/img2img"
        logger.info(f"[sdWebui] POST {url}  payload={{略}}")
        async with httpx.AsyncClient(timeout=self.timeout) as cli:
            resp = await cli.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        images = data.get("images", [])
        if not images:
            raise RuntimeError("WebUI 未返回 images")
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        paths: List[Path] = []
        for idx, b64 in enumerate(images, start=1):
            if "," in b64:
                b64 = b64.split(",", 1)[1]
            img_bytes = base64.b64decode(b64)
            file_path = self.img_dir / f"{ts}_img2img_{idx}.png"
            file_path.write_bytes(img_bytes)
            paths.append(file_path)
        logger.info(f"[sdWebui] 保存 {len(paths)} 张图片 → {self.img_dir}")
        return paths

    # ────────────── extras 单图超分 ──────────────
    async def upscaling(self, image_bytes: bytes, upscaler: str = None, scale: float = None) -> Path:
        payload = {
            "image": base64.b64encode(image_bytes).decode(),
            "upscaling_resize": scale or self.upscale_factor,
            "upscaler_1": upscaler or self.upscaler or "Latent",
            "resize_mode": 0,
            "show_extras_results": True
        }
        url = f"{self.base_url}/sdapi/v1/extra-single-image"
        async with httpx.AsyncClient(timeout=self.timeout) as cli:
            resp = await cli.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        b64 = data.get("image", "")
        if not b64:
            raise RuntimeError("WebUI未返回超分结果")
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        img_bytes = base64.b64decode(b64)
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        file_path = self.img_dir / f"{ts}_upscale.png"
        file_path.write_bytes(img_bytes)
        logger.info(f"[sdWebui] 保存超分图片 → {file_path}")
        return file_path

    # ────────────── 并发/信号量  ──────────────
    async def run_with_limit(self, coro):
        async with self.task_semaphore:
            return await coro

    # ────────────── /sd txt2img ──────────────
    @filter.command("sd")
    async def sd_cmd(self, event: AstrMessageEvent):
        prompt_raw = self._extract_plain(event.message_obj.message)
        if not prompt_raw:
            yield event.plain_result("❌ 请输入提示词。例如 /sd cyberpunk city")
            return
        prompt = prompt_raw.strip()
        if self.auto_tr:
            prompt = await self.translate_prompt(prompt)
        if self.llm_gen:
            prompt = await self.llm_generate_prompt(prompt)
        pm = self.merge_prompt(prompt)
        if self.show_positive:
            yield event.plain_result(f"正向提示词：{pm['positive']}")
        try:
            paths = await self.run_with_limit(self.txt2img(
                pm['positive'], pm['negative']
            ))
        except Exception as e:
            logger.warning(f"[sdWebui] 生成失败：{e}")
            yield event.plain_result(f"❌ 图片生成失败：{e}")
            return
        chain: List[Comp.MessageSegment] = [
            Comp.At(qq=event.get_sender_id()),
            Comp.Plain(f"已生成 {len(paths)} 张图片\n提示词: {prompt}")
        ]
        for p in paths:
            chain.append(build_image_component(p))
        yield event.chain_result(chain)

    # ────────────── /sdi img2img ──────────────
    @filter.command("sdi")
    async def sdi_cmd(self, event: AstrMessageEvent):
        # 取图片与提示词
        img_bytes = None
        prompt_raw = None
        for seg in event.message_obj.message:
            if seg.type == "Image":
                img_bytes = await self.ctx.download_image(seg.url)
            elif seg.type == "Plain":
                prompt_raw = re.sub(r"^/sdi\s*", "", seg.text, count=1)
        if not img_bytes or not prompt_raw:
            yield event.plain_result("❌ 请发送图片和提示词，例如 /sdi cyberpunk city + 图片")
            return
        prompt = prompt_raw.strip()
        if self.auto_tr:
            prompt = await self.translate_prompt(prompt)
        if self.llm_gen:
            prompt = await self.llm_generate_prompt(prompt)
        pm = self.merge_prompt(prompt)
        if self.show_positive:
            yield event.plain_result(f"正向提示词：{pm['positive']}")
        try:
            paths = await self.run_with_limit(self.img2img(
                img_bytes, pm['positive'], pm['negative']
            ))
        except Exception as e:
            logger.warning(f"[sdWebui] img2img失败：{e}")
            yield event.plain_result(f"❌ 图片生成失败：{e}")
            return
        chain: List[Comp.MessageSegment] = [
            Comp.At(qq=event.get_sender_id()),
            Comp.Plain(f"img2img已生成 {len(paths)} 张图片\n提示词: {prompt}")
        ]
        for p in paths:
            chain.append(build_image_component(p))
        yield event.chain_result(chain)

    # ────────────── /sd model list/set ──────────────
    @filter.command("sd_model")
    async def sd_model(self, event: AstrMessageEvent):
        """列出/切换模型 /sd_model [list|set n]"""
        msg = self._extract_plain(event.message_obj.message) or ""
        if msg.startswith("list"):
            try:
                self._models = await self.fetch_webui_resource("model")
                text = "\n".join([f"{i+1}. {m}" for i, m in enumerate(self._models)])
                yield event.plain_result(f"可用模型：\n{text}")
            except Exception as e:
                yield event.plain_result(f"❌ 获取模型失败：{e}")
        elif msg.startswith("set"):
            try:
                idx = int(msg.split()[1]) - 1
                model = self._models[idx]
                # 热切换
                async with httpx.AsyncClient(timeout=self.timeout) as cli:
                    resp = await cli.post(
                        f"{self.base_url}/sdapi/v1/options",
                        json={"sd_model_checkpoint": model}
                    )
                    resp.raise_for_status()
                yield event.plain_result(f"已切换模型为：{model}")
            except Exception as e:
                yield event.plain_result(f"❌ 切换模型失败：{e}")

    # ────────────── /sd_sampler ──────────────
    @filter.command("sd_sampler")
    async def sd_sampler(self, event: AstrMessageEvent):
        """采样器管理 /sd_sampler [list|set n]"""
        msg = self._extract_plain(event.message_obj.message) or ""
        if msg.startswith("list"):
            self._samplers = await self.fetch_webui_resource("sampler")
            text = "\n".join([f"{i+1}. {m}" for i, m in enumerate(self._samplers)])
            yield event.plain_result(f"可用采样器：\n{text}")
        elif msg.startswith("set"):
            try:
                idx = int(msg.split()[1]) - 1
                sampler = self._samplers[idx]
                self.sampler = sampler
                self.config["sampler"] = sampler
                yield event.plain_result(f"采样器已设置为：{sampler}")
            except Exception as e:
                yield event.plain_result(f"❌ 切换采样器失败：{e}")

    # ────────────── /sd_upscaler ──────────────
    @filter.command("sd_upscaler")
    async def sd_upscaler(self, event: AstrMessageEvent):
        msg = self._extract_plain(event.message_obj.message) or ""
        if msg.startswith("list"):
            self._upscalers = await self.fetch_webui_resource("upscaler")
            text = "\n".join([f"{i+1}. {m}" for i, m in enumerate(self._upscalers)])
            yield event.plain_result(f"可用超分算法：\n{text}")
        elif msg.startswith("set"):
            try:
                idx = int(msg.split()[1]) - 1
                upscaler = self._upscalers[idx]
                self.upscaler = upscaler
                self.config["upscaler"] = upscaler
                yield event.plain_result(f"超分算法已设置为：{upscaler}")
            except Exception as e:
                yield event.plain_result(f"❌ 切换超分失败：{e}")

    # ────────────── /sd_lora ──────────────
    @filter.command("sd_lora")
    async def sd_lora(self, event: AstrMessageEvent):
        """LoRA管理 /sd_lora [list|set n weight]"""
        msg = self._extract_plain(event.message_obj.message) or ""
        if msg.startswith("list"):
            self._lora = await self.fetch_webui_resource("lora")
            text = "\n".join([f"{i+1}. {m}" for i, m in enumerate(self._lora)])
            yield event.plain_result(f"可用LoRA：\n{text}")
        elif msg.startswith("set"):
            try:
                idx, weight = int(msg.split()[1]) - 1, float(msg.split()[2])
                lora_name = self._lora[idx]
                self._lora_weights[lora_name] = weight
                yield event.plain_result(f"LoRA已设置：{lora_name}，权重：{weight}")
            except Exception as e:
                yield event.plain_result(f"❌ LoRA设置失败：{e}")

    # ────────────── /sd_embedding ──────────────
    @filter.command("sd_embedding")
    async def sd_embedding(self, event: AstrMessageEvent):
        self._embedding = await self.fetch_webui_resource("embedding")
        text = "\n".join([f"{i+1}. {m}" for i, m in enumerate(self._embedding)])
        yield event.plain_result(f"可用Embedding：\n{text}")

    # ────────────── /sd_conf 打印当前所有参数 ──────────────
    @filter.command("sd_conf")
    async def sd_conf(self, event: AstrMessageEvent):
        info = (
            f"当前模型: {self.config.get('default_model', '')}\n"
            f"采样器: {self.sampler}\n"
            f"分辨率: {self.w}x{self.h}\n"
            f"步数: {self.steps}\n"
            f"CFG: {self.cfg}\n"
            f"批量: {self.batch}\n"
            f"迭代: {self.n_iter}\n"
            f"全局正向: {self.positive_prompt}\n"
            f"全局负向: {self.negative_prompt}\n"
            f"超分算法: {self.upscaler}\n"
            f"超分倍数: {self.upscale_factor}\n"
            f"HiRes Fix: {self.hires_fix}\n"
            f"最大并发: {self.max_tasks}\n"
            f"请求超时: {self.timeout}\n"
            f"详细模式: {self.verbose}\n"
        )
        yield event.plain_result(info)

    # ────────────── /sd_help ──────────────
    @filter.command("sd_help")
    async def sd_help(self, event: AstrMessageEvent):
        help_text = (
            "🖼️ Stable Diffusion WebUI 全功能插件 指令指南\n"
            "生成图像: /sd 提示词\n"
            "图生图: /sdi 提示词+图片\n"
            "模型管理: /sd_model list | set n\n"
            "采样器管理: /sd_sampler list | set n\n"
            "超分算法: /sd_upscaler list | set n\n"
            "LoRA: /sd_lora list | set n weight\n"
            "Embedding: /sd_embedding\n"
            "参数总览: /sd_conf\n"
            "帮助: /sd_help\n"
            "⚙️ 详细功能请查阅文档"
        )
        yield event.plain_result(help_text)

    # ────────────── 工具 ──────────────
    @staticmethod
    def _extract_plain(msg_list) -> str | None:
        for seg in msg_list:
            if seg.type == "Plain":
                return seg.text
        return None
