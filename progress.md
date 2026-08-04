# G1 MuJoCo 强化学习进度

<!-- AUTO_STATUS_START -->
## 实时状态

- 自动同步时间：尚未运行
- 当前阶段：尚未开始
- 最近训练步数：尚无记录 / 5000000
- 最近吞吐：尚无记录 steps/s
- Rollout 平均奖励：尚无记录
- Rollout 平均回合长度：尚无记录 个控制步
- 最近独立评估：尚未生成
<!-- AUTO_STATUS_END -->

## 使用方式

手动同步一次：

```bash
python3 progress_sync.py
```

每十分钟持续同步：

```bash
python3 progress_sync.py --watch --interval 600
```

该文件只记录本机训练状态，不应提交模型权重、日志或个人环境信息。
