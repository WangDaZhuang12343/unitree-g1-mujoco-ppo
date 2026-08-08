# Codex 登录与新电脑配置指南

本文说明 Codex 网页版、终端和 VS Code 的登录区别，以及如何在受信任的新电脑上复用现有配置。命令中的用户名、IP 和密钥均需替换为实际值。

## 1. 登录方式对照

| 使用方式 | 登录方式 | 是否可仅用 API Key | 说明 |
|---|---|---:|---|
| ChatGPT/Codex 网页版 | ChatGPT 账号 | 否 | Codex 云端需要 ChatGPT 账号及相应权限 |
| Codex CLI | ChatGPT 账号或 API Key | 是 | API Key 按 API 用量计费 |
| VS Code Codex 扩展 | ChatGPT 账号或 API Key | 是 | 通常与 CLI 共用本机登录缓存 |
| Codex 云端环境中的 Secrets | 已登录后配置环境变量 | 不适用 | Key 供项目运行，不能代替网页登录 |

注意：网页中出现的 API Key 或 Secrets 输入框，通常用于把密钥注入云端任务的运行环境，不是网页版账号登录入口。

## 2. 网页版登录

1. 打开 <https://chatgpt.com>。
2. 使用有 Codex 权限的 ChatGPT 账号登录并选择正确的工作区。
3. 如需操作 GitHub 项目，在 Codex 中授权 GitHub，并只开放所需仓库。
4. 建议为账号启用多因素认证（MFA）。

网页版不能通过复制 `~/.codex/auth.json` 或填写 Platform API Key 代替 ChatGPT 登录。账号、网络和所在地区必须满足 OpenAI 的服务要求。

## 3. 新电脑安装 Codex CLI

无需 `sudo` 的安装方式：

```bash
npm install -g --prefix ~/.local @openai/codex@latest --include=optional
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.profile
source ~/.profile
codex --version
```

使用 API Key 登录时，不要把 Key 直接写进命令历史：

```bash
read -rsp "OpenAI API Key: " OPENAI_API_KEY
echo
export OPENAI_API_KEY
printenv OPENAI_API_KEY | codex login --with-api-key
unset OPENAI_API_KEY
codex login status
```

如果使用公司自建模型代理，还需要沿用公司提供的 `model_provider`、Base URL 和认证方式；OpenAI Platform Key 与公司代理 Key 不应混用。

## 4. VS Code 登录

1. 安装官方扩展 `openai.chatgpt`。
2. 执行 `Developer: Reload Window`。
3. 在扩展登录页选择 **Use API Key**，或使用已由 CLI 缓存的本机凭据。
4. 用终端检查：

```bash
codex login status
code --list-extensions --show-versions | grep '^openai.chatgpt@'
```

CLI 正常但扩展未登录时，先重载 VS Code；仍失败则退出扩展账号后重新选择 **Use API Key**。

## 5. 从旧电脑迁移到受信任的新电脑

优先在新电脑使用一枚独立、可撤销的 API Key。确需迁移现有配置时，可通过 SSH 传输：

```bash
ssh-copy-id USER@HOST
ssh USER@HOST 'mkdir -p ~/.codex && chmod 700 ~/.codex'
scp ~/.codex/config.toml USER@HOST:~/.codex/config.toml
scp ~/.codex/auth.json USER@HOST:~/.codex/auth.json
ssh USER@HOST 'chmod 600 ~/.codex/config.toml ~/.codex/auth.json'
```

然后在新电脑验证：

```bash
codex login status
codex --version
```

`auth.json` 相当于密码，只能传到本人控制且可信的设备。若系统使用操作系统密钥环而不是文件存储，此迁移方法可能不适用，应在新电脑重新登录。

## 6. 公司 API Key 的使用边界

- 先确认公司允许该 Key 用于 OpenAI/Codex，且允许提交给云端运行环境。
- 推荐新建用途单一、权限受限、可轮换和可撤销的专用 Key。
- 不要把共享主 Key 填入第三方网页。
- 不要把 Key 写进 Git、README、截图、聊天记录或 Shell 历史。
- 网页 Secrets 中的 Key 仅供任务运行，不能解锁 Codex 网页版。
- 自建代理 Key 往往还依赖专用 Base URL，不能假定兼容 OpenAI 官方入口。

## 7. 安全检查

```bash
chmod 700 ~/.codex
chmod 600 ~/.codex/config.toml ~/.codex/auth.json
git status --short
```

确认仓库未包含以下内容：

```text
.codex/auth.json
OPENAI_API_KEY=真实密钥
公司代理真实密钥
```

如密钥曾出现在公开位置、Git 历史或聊天中，应立即撤销并重新生成。

## 8. 官方参考

- [Codex Authentication](https://learn.chatgpt.com/docs/auth)
- [ChatGPT](https://chatgpt.com)
- [OpenAI API Keys](https://platform.openai.com/api-keys)

