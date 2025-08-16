# SDWebuiApi 插件

> **Stable Diffusion 本地 WebUI × AstrBot**  
> 零依赖图床，生成图片落地本机、自动上传到 QQ / 频道。  

---

## 功能一览

| 功能                  | 说明                                                         |
| --------------------- | ------------------------------------------------------------ |
| `/sd <提示词>`        | 调用 **本地** Stable Diffusion WebUI `/sdapi/v1/txt2img` 直接出图 |
| `/sdi <提示词> +图片` | 图生图（img2img），支持图片引导与批量生成                    |
| 自动保存              | 生成图片按时间戳保存在 `plugins/astrbot_plugin_sdwebuiapi/images/` |
| 自动发送              | Bot 会自动识别可用的本地图片构造器并上传；无须 OSS           |
| 多张生成              | `batch`/`n_iter` 可配置，一次最多返回 8 张（按 WebUI 配置）  |
| Hi-Res Fix            | 支持 txt2img 内部高清放大，提升细节                          |
| 超分辨率              | `/sd_extras` 指令支持图像单独超分（WebUI Extras接口）        |
| 模型热切换            | `/sd_model list/set` 一键切换基础模型                        |
| LoRA/Embedding        | `/sd_lora list/set` `/sd_embedding` 支持 LoRA 选择/权重/Embedding |
| Sampler/Upscaler      | `/sd_sampler`、`/sd_upscaler` 列表/切换                      |
| ControlNet            | 预留接口，兼容 WebUI 的 additional networks                  |
| 全局负向提示词        | 所有出图自动拼接自定义负向 prompt                            |
| LLM自动提示词         | 支持 Bot LLM 生成/润色正向 prompt                            |
| 中文翻译              | `translate_prompt=true` 时，中文 prompt 自动转英文           |
| 并发控制              | 最大并发信号量、任务队列，支持多任务高并发                   |
| 动态参数热切          | 分辨率/步数/batch/n_iter/采样器/超分等全参数指令热切         |
| verbose模式           | 详细日志、超时自定义、正向 prompt 回显                       |
| 配置面板热更新        | 所有参数支持控制台实时编辑                                   |
| `/sd_help` `/sd_conf` | 显示所有支持能力与当前参数                                   |

------

## 环境要求

| 依赖                       | 最低版本                                                  |
| -------------------------- | --------------------------------------------------------- |
| **AstrBot**                | ≥ 0.9（建议 0.9.2+，支持本地图片上传）                    |
| **Python**                 | ≥ 3.8                                                     |
| Stable Diffusion **WebUI** | 任意版本，需 **启动时带 `--api`**                         |
| **pip 依赖**               | `httpx>=0.25.0`（插件会自动通过 `requirements.txt` 安装） |

------

## 安装步骤

```bash
# 1. 克隆 / 下载
cd data/plugins/
git clone https://github.com/orran/astrbot_plugin_sdwebuiapi.git
# 或手动解压到此目录

# 2. 重载或重启 AstrBot
/reload_plugins
```

> **首次启动** 会自动安装 `requirements.txt` 中的依赖，请保持外网或本地 PyPI 镜像畅通。

------

## 配置说明

打开 **控制台 → 插件管理 → sdWebuiApi → 配置**，可见以下字段（核心参数，详细参数见后续“指令/参数表”）：

| 字段                     | 说明                       | 默认                    |
| ------------------------ | -------------------------- | ----------------------- |
| `base_url`               | WebUI 接口地址             | `http://127.0.0.1:7860` |
| `output_dir`             | 本地保存目录（相对插件根） | `images`                |
| `width` / `height`       | 分辨率                     | `512 / 512`             |
| `steps`                  | 采样步数                   | `20`                    |
| `cfg`                    | CFG Scale                  | `7.0`                   |
| `sampler`                | 采样器名称                 | `Euler a`               |
| `batch` / `n_iter`       | 一次生成张数/迭代次数      | `1` / `1`               |
| `hires_fix`              | txt2img高清放大            | `false`                 |
| `upscaler`               | 超分算法                   | `None`（按 WebUI 支持） |
| `upscale_factor`         | 超分倍数                   | `2.0`                   |
| `translate_prompt`       | 自动翻译中文提示词         | `false`                 |
| `llm_generate_prompt`    | LLM 智能生成 prompt        | `false`                 |
| `negative_prompt_global` | 全局负向 prompt            | `"bad, blurry, ..."`    |
| `max_concurrent_tasks`   | 最大并发任务               | `3`                     |
| `session_timeout`        | 每次API调用超时（秒）      | `180`                   |
| `show_positive_prompt`   | 回显正向 prompt            | `false`                 |
| `verbose`                | 详细日志模式               | `true`                  |



修改后点击 **保存**，无需重启。

------

## 指令与参数总览

| 指令                    | 说明                                           | 示例                     |
| ----------------------- | ---------------------------------------------- | ------------------------ |
| `/sd <提示词>`          | 文本生图（txt2img），支持批量、负向、翻译、LLM | `/sd 赛博朋克女孩，高清` |
| `/sdi <提示词> +图片`   | 图生图（img2img），支持内容/分辨率等同步配置   | `/sdi 魔法少女 + [图片]` |
| `/sd_model list/set`    | 列表/切换 WebUI 现有模型                       | `/sd_model set 2`        |
| `/sd_lora list/set`     | LoRA模型管理，权重可调                         | `/sd_lora set 1 0.8`     |
| `/sd_embedding`         | Embedding 列表                                 | `/sd_embedding`          |
| `/sd_sampler list/set`  | 列出/切换采样器                                | `/sd_sampler set 3`      |
| `/sd_upscaler list/set` | 列出/切换超分算法                              | `/sd_upscaler set 1`     |
| `/sd_extras`            | Extras 超分/人脸修复                           | `/sd_extras [图片]`      |
| `/sd_conf`              | 打印当前所有配置参数                           |                          |
| `/sd_help`              | 显示本插件详细帮助                             |                          |
| `/sd_verbose`           | 开关详细输出                                   |                          |
| `/sd_batch <数量>`      | 设置生成批次（一次多张）                       | `/sd_batch 4`            |
| `/sd_step <步数>`       | 设置生成步数                                   | `/sd_step 30`            |
| `/sd_res <高> <宽>`     | 设置分辨率                                     | `/sd_res 1024 768`       |
| `/sd_iter <n>`          | 设置迭代次数                                   | `/sd_iter 3`             |
| `/sd_timeout <秒>`      | 修改每次API超时                                | `/sd_timeout 180`        |



> 所有参数可随时 `/sd_conf` 查看当前状态。
>
> `/sd_model`、`/sd_sampler`、`/sd_lora`、`/sd_embedding` 等均支持**动态资源刷新和热切换**，完全贴合 WebUI 实时状态。

------

## 使用示例

```bash
/sd cyberpunk city at dusk, ultra detailed, 4k
```

Bot 自动返回：

- `@你`
- 生成的 1～N 张图片（本地文件，极速上传，无需图床）
- 文本：`已生成 3 张图片\n提示词: ...`

> 若启用 `translate_prompt=true` 并输入中文：
>
> ```bash
> /sd 一个拿着激光剑的女孩，霓虹灯
> ```
>
> 插件会先通过 Bot 的 LLM 翻译成英文，再送 WebUI，结果全程回显。

------

## 二次开发

- **增加新指令**
   仿照 `@filter.command("sd")` 新增指令与处理方法，参数、API结构高度一致。
- **图生图**
   增加 `/sdi` 指令，对接 `/sdapi/v1/img2img`，参数参考 Swagger 文档。
- **自定义批量/采样/超分/ControlNet**
   直接扩展 `main.py`，所有资源热加载与指令解析可自定义。
- **API能力扩展**
   支持全 WebUI API，如 ControlNet、脚本注入、自定义回调等。

------

## 致谢

- [Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
- [AstrBot](https://github.com/AstrBotDevs/AstrBot)

------

欢迎 PR / Issue
