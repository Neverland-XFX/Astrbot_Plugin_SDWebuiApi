# astrbot_plugin_sdwebuiapi/main.py
from __future__ import annotations
import base64, datetime as dt, uuid, re
from pathlib import Path
from typing import List

import httpx
from colorama import Fore, Style, init as colorama_init

from astrbot.api.star import Star, register, Context
from astrbot.api.event import filter, AstrMessageEvent
import astrbot.api.message_components as Comp
from astrbot.api import logger


# ─────────────────────────── 工具：本地图片 → Comp.Image ────────────────────────────
def build_image_component(filepath: Path) -> Comp.MessageSegment:
    """
    根据 AstrBot 运行版本动态选择可用的本地图片构造器。
    若全部缺失则降级为 Base64（几乎所有平台可用）；再不行就 Plain 文本。
    """
    path_str = str(filepath.resolve())

    if hasattr(Comp.Image, "fromPath"):
        return Comp.Image.fromPath(path_str)
    if hasattr(Comp.Image, "fromFile"):
        return Comp.Image.fromFile(path_str)
    if hasattr(Comp.Image, "fromFilePath"):
        return Comp.Image.fromFilePath(path_str)

    # ↓ 没有本地构造器时的兜底方案
    try:
        b64 = base64.b64encode(filepath.read_bytes()).decode()
        if hasattr(Comp.Image, "fromBase64"):
            return Comp.Image.fromBase64(b64)
        if hasattr(Comp.Image, "fromBytes"):
            return Comp.Image.fromBytes(base64.b64decode(b64), f"{uuid.uuid4()}.png")
    except Exception as e:
        logger.warning(f"[sdWebui] Base64 兜底失败：{e}")

    # 最后再兜底为纯文本
    return Comp.Plain(f"[图片保存在本地 {filepath}] (当前平台无本地图片发送能力)")


# ─────────────────────────── 主插件 ───────────────────────────
@register(
    "sdWebuiApi",
    "orran",
    "调用本地 Stable Diffusion WebUI 生成图片",
    "0.1.0"
)
class SdWebuiApi(Star):
    def __init__(self, context: Context, config: dict, *_):
        colorama_init()  # 让本地日志彩色
        self.ctx: Context = context

        # 读取配置
        self.base_url = config.get("base_url", "http://127.0.0.1:7860").rstrip("/")
        img_dir_name  = config.get("output_dir", "images")
        self.img_dir  = (Path(__file__).parent / img_dir_name).resolve()
        self.w        = int(config.get("width", 512))
        self.h        = int(config.get("height", 512))
        self.steps    = int(config.get("steps", 20))
        self.cfg      = float(config.get("cfg", 7.0))
        self.sampler  = str(config.get("sampler", "Euler a"))
        self.batch    = int(config.get("batch", 1))
        self.auto_tr  = bool(config.get("translate_prompt", False))

        self.img_dir.mkdir(parents=True, exist_ok=True)
        super().__init__(context)

    # ─────────────────────────── 辅助 — 翻译 (可关闭) ───────────────────────────
    async def translate_prompt(self, text: str) -> str:
        if not self.auto_tr or not any("\u4e00" <= ch <= "\u9fff" for ch in text):
            return text
        rsp = await self.ctx.get_using_provider().text_chat(prompt=text)
        translated = rsp.result_chain.chain[0].text.strip()
        logger.info(f"[sdWebui] 翻译：{text} → {translated}")
        return translated or text

    # ─────────────────────────── 调 WebUI ───────────────────────────
    async def txt2img(self, prompt: str) -> List[Path]:
        payload = {
            "prompt": prompt,
            "steps": self.steps,
            "cfg_scale": self.cfg,
            "sampler_index": self.sampler,
            "width": self.w,
            "height": self.h,
            "batch_size": self.batch,
            "seed": -1
        }
        url = f"{self.base_url}/sdapi/v1/txt2img"
        logger.info(f"[sdWebui] POST {url}  payload={payload}")
        async with httpx.AsyncClient(timeout=300) as cli:
            resp = await cli.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

        images = data.get("images", [])
        if not images:
            raise RuntimeError("WebUI 未返回 images")

        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        paths: List[Path] = []
        for idx, b64 in enumerate(images, start=1):
            if "," in b64:  # strip data:image/png;base64,
                b64 = b64.split(",", 1)[1]
            img_bytes = base64.b64decode(b64)
            file_path = self.img_dir / f"{ts}_{idx}.png"
            file_path.write_bytes(img_bytes)
            paths.append(file_path)

        logger.info(f"[sdWebui] 保存 {len(paths)} 张图片 → {self.img_dir}")
        return paths

    # ─────────────────────────── 指令 /sd … ───────────────────────────
    @filter.command("sd")
    async def sd_cmd(self, event: AstrMessageEvent):
        prompt_raw = self._extract_plain(event.message_obj.message)
        if not prompt_raw:
            yield event.plain_result("❌ 请输入提示词。例如 /sd cyberpunk city")
            return

        prompt = prompt_raw.strip()
        prompt = await self.translate_prompt(prompt)

        try:
            paths = await self.txt2img(prompt)
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

    # ─────────────────────────── 工具：取首段 Plain 文本 ───────────────────────────
    @staticmethod
    def _extract_plain(msg_list) -> str | None:
        for seg in msg_list:
            if seg.type == "Plain":
                # 去掉指令前缀  `/sd `
                return re.sub(r"^/sd\s*", "", seg.text, count=1)
        return None
