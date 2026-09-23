**读**

- README「Install Samples」和「Running All Samples as Tests」
- `run_tests.py` 前 120 行：怎么找可执行文件、怎么匹配 `test_args.json`
- `test_args.json` 里三种形态各找一个例子：`skip`、单次 `args`、多次 `runs`（README 用 `fluidsGL`、`ptxgen`、`recursiveGaussian` 举例）

**做**

- 不要在本周跑完全部测试。只对周二编出的 `deviceQuery` 做一次：`python run_tests.py --dir <该可执行文件所在目录> --config test_args.json`
- 记下安装路径模板：`build/bin/${TARGET_ARCH}/${TARGET_OS}/${BUILD_TYPE}`，Windows 例子是 `build/bin/amd64/windows/release`

**想清楚**

1. README 为什么强调这些样例不是 CUDA 的验证套件？
2. 图形样例为什么经常标 `skip`？
3. 找不到 JSON 条目时，脚本会传什么参数？

**完成标准**：能解释一次测试运行的退出码含义（0 成功、非 0 失败、样例自己 waive）。