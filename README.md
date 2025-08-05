# SDWebuiApi 插件

> **Stable Diffusion 本地 WebUI × AstrBot**  
> 零依赖图床，生成图片落地本机、自动上传到 QQ / 频道。  

---

## 功能一览

| 功能           | 说明                                                         |
| -------------- | ------------------------------------------------------------ |
| `/sd <提示词>` | 调用 **本地** Stable Diffusion WebUI `/sdapi/v1/txt2img` 直接出图 |
| 自动保存       | 生成图片按时间戳保存在 `plugins/astrbot_plugin_sdwebuiapi/images/` |
| 自动发送       | Bot 会自动识别可用的本地图片构造器并上传；无须 OSS           |
| 多张生成       | `batch` 可配置，一次最多返回 8 张（按 WebUI 配置）           |
| 自动翻译       | `translate_prompt=true` 时，检测到中文提示词先走 LLM 翻译    |
| 可配项丰富     | 分辨率 / 步数 / CFG / 采样器 / WebUI 地址 / 输出目录…全可在面板修改 |

---

## 环境要求

| 依赖                       | 最低版本                                                     |
| -------------------------- | ------------------------------------------------------------ |
| **AstrBot**                | ≥ 0.9（建议 0.9.2+，支持本地图片上传）                       |
| **Python**                 | ≥ 3.8                                                        |
| Stable Diffusion **WebUI** | 任意版本，需 **启动时带 `--api`**                            |
| **pip 依赖**               | `httpx>=0.25.0`、`colorama>=0.4.6`（插件会自动通过 `requirements.txt` 安装） |

---

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

打开 **控制台 → 插件管理 → sdWebuiApi → 配置**，可见以下字段：

| 字段               | 说明                       | 默认                    |
| ------------------ | -------------------------- | ----------------------- |
| `base_url`         | WebUI 接口地址             | `http://127.0.0.1:7860` |
| `output_dir`       | 本地保存目录（相对插件根） | `images`                |
| `width` / `height` | 分辨率                     | `512 / 512`             |
| `steps`            | 采样步数                   | `20`                    |
| `cfg`              | CFG Scale                  | `7.0`                   |
| `sampler`          | 采样器名称                 | `Euler a`               |
| `batch`            | 一次生成张数               | `1`                     |
| `translate_prompt` | 自动翻译中文提示词         | `false`                 |

修改后点击 **保存**，无需重启。

------

## 使用示例

```bash
/sd cyberpunk city at dusk, ultra detailed, 4k
```

Bot 返回：

- `@你`
- 生成的 1～N 张图片
- 文本：`已生成 3 张图片\n提示词: cyberpunk city at dusk ...`

> 若启用 `translate_prompt=true` 并输入中文：
>
> ```bash
> /sd 一个拿着激光剑的女孩，霓虹灯
> ```
>
> 插件会先通过 Bot 的默认 LLM 翻译成英文，再投递给 WebUI。

------

## 常见问题

| Q                           | A                                                            |
| --------------------------- | ------------------------------------------------------------ |
| **机器人发不出图片？**      | 90% 是当前平台或 AstrBot 版本缺乏“本地上传”API。插件已自动 fallback 到 Base64；若仍失败请升级 AstrBot 或切换图床。 |
| **WebUI 404 / 连接拒绝？**  | 确认 WebUI 启动时加 `--api`，并与 `base_url` 端口匹配。      |
| **想换模型 / ControlNet？** | 目前插件只包了最基础的 `txt2img`。可在 `main.py` 的 `txt2img()` 里追加 WebUI 参数，或先 `POST /sdapi/v1/options` 切模型，再生成。 |
| **磁盘占用过大？**          | 定时清理 `output_dir`，或在发送完图片后自行 `Path.unlink()`。 |

------

## 二次开发指北

- **修改指令名**

  ```python
  @filter.command("sd")  # → 改成你想要的别名
  ```

- **添加 `/img2img`**

  - 新增函数 `async def sd_img2img(...)`
  - 调用 `/sdapi/v1/img2img`，构造体参考 Swagger (`http://<webui>/docs`)
  - 参考 `sd_cmd` 注册新指令

- **动态切模型**

  ```python
  async with httpx.AsyncClient() as cli:
      await cli.post(f"{base_url}/sdapi/v1/options",
                     json={"sd_model_checkpoint": "anythingV5.safetensors"})
  ```

欢迎提 PR / Issue！