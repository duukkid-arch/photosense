import os
from dotenv import load_dotenv
from dashscope import Generation
from http import HTTPStatus

# 从 .env 加载 API Key
load_dotenv()
api_key = os.getenv('DASHSCOPE_API_KEY')

# 检查 Key 格式
if not api_key:
    print('❌ DASHSCOPE_API_KEY 没有读取到')
    exit(1)

if api_key.startswith('sk-sk-'):
    print(f'❌ Key 格式错误!检测到 sk-sk- 重复前缀')
    print(f'   当前: {api_key[:15]}...')
    print(f'   请编辑 .env 删掉多余的 sk-')
    exit(1)

if not api_key.startswith('sk-'):
    print(f'❌ Key 应该以 sk- 开头')
    exit(1)

print(f'✅ API Key 格式正确 ({api_key[:8]}...)')
print('正在调用 Qwen API...')
print()

# 调用 API
response = Generation.call(
    api_key=api_key,
    model='qwen-turbo',
    messages=[{'role': 'user', 'content': '你好,用一句话介绍 PhotoSense 项目可能是什么'}]
)

# 检查响应
if response.status_code != HTTPStatus.OK:
    print(f'❌ API 调用失败')
    print(f'   状态码: {response.status_code}')
    print(f'   错误码: {response.code}')
    print(f'   错误信息: {response.message}')
    exit(1)

print('=== Qwen 回复 ===')
print(response.output.text)
print()
print('✅ API 测试成功!环境完全 OK!')
