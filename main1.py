# 导入操作系统相关模块，用于文件操作、清屏等
import os
# 导入JSON模块，用于对话历史的读写
import json
# 导入正则表达式模块，用于匹配代码块和文件引用
import re
# 导入异步IO模块，实现非阻塞的API调用
import asyncio
# 导入dotenv，用于加载.env环境变量
from dotenv import load_dotenv
# 导入prompt_toolkit相关组件，构建交互式命令行界面
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.styles import Style
# 导入pygments，用于代码高亮
from pygments import highlight
from pygments.lexers import get_lexer_by_name, guess_lexer
from pygments.formatters import TerminalFormatter
# 导入tiktoken，用于计算token数量
import tiktoken

# 导入OpenAI异步客户端（兼容DeepSeek、通义千问等OpenAI格式API）
from openai import AsyncOpenAI
# 尝试导入新版Google Gemini SDK，失败则兼容旧版
try:
    import google.genai as genai
except:
    import google.generativeai as genai
# 导入Anthropic Claude异步客户端
from anthropic import AsyncAnthropic

# 加载.env文件中的环境变量
load_dotenv()

# 全局配置字典
config = {
    'ctx_win': int(os.getenv('DEFAULT_CONTEXT_WINDOW', 8192)),  # 上下文窗口大小
    'hist_file': 'chat_history.json',  # 对话历史保存文件
    'input_hist': '.input_history',  # 输入历史保存文件
    'reserve_token': 1024,  # 为AI回复预留的token数
    'default_api': os.getenv('DEFAULT_API_SOURCE', 'deepseek'),  # 默认使用的API
}

# 命令行界面样式配置
cli_style = Style.from_dict({
    'user-prompt': '#66ff66 bold',  # 用户提示样式
    'ai-prompt': '#3399ff bold',     # AI提示样式
    'system': '#ffcc00 italic',       # 系统信息样式
    'error': '#ff4444 bold',          # 错误信息样式
    'success': '#00ff99 bold',        # 成功信息样式
})

# 各API平台配置
api_sources = {
    'deepseek': {
        'name': 'DeepSeek',
        'key': os.getenv('DEEPSEEK_API_KEY'),  # 从环境变量获取API密钥
        'url': 'https://api.deepseek.com/v1',   # API地址
        'def_model': 'deepseek-chat',            # 默认模型
        'models': ['deepseek-chat', 'deepseek-coder-v2']  # 支持的模型列表
    },
    'qwen': {
        'name': '通义千问',
        'key': os.getenv('QWEN_API_KEY'),
        'url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        'def_model': 'qwen-turbo',
        'models': ['qwen-turbo','qwen-plus','qwen-max','qwen-coder']
    },
    'openai': {
        'name': 'OpenAI',
        'key': os.getenv('OPENAI_API_KEY'),
        'url': 'https://api.openai.com/v1',
        'def_model': 'gpt-3.5-turbo',
        'models': ['gpt-3.5-turbo', 'gpt-4o']
    },
    'gemini': {
        'name': 'Google Gemini',
        'key': os.getenv('GOOGLE_API_KEY'),
        'url': '',  # Gemini无需base_url
        'def_model': 'gemini-1.5-flash',
        'models': ['gemini-1.5-flash', 'gemini-1.5-pro']
    },
    'claude': {
        'name': 'Anthropic Claude',
        'key': os.getenv('ANTHROPIC_API_KEY'),
        'url': '',  # Claude无需base_url
        'def_model': 'claude-3-5-sonnet',
        'models': ['claude-3-5-sonnet', 'claude-3-opus']
    }
}

# Token管理类：计算token数量、截断超长对话历史
class TokenManager:
    def __init__(self, model='gpt-3.5-turbo'):
        self.encoder = tiktoken.get_encoding('cl100k_base')  # 使用cl100k_base编码
        self.model = model

    # 计算单段文本的token数
    def get_tokens(self, text):
        return len(self.encoder.encode(text))

    # 计算对话历史的总token数
    def get_msgs_tokens(self, msgs):
        total = 0
        for m in msgs:
            total += self.get_tokens(m['content'])
            total += self.get_tokens(m['role'])
        return total

    # 截断对话历史，保留系统提示和最新对话
    def cut_history(self, msgs, max_tok, keep_sys=True):
        if not msgs:
            return msgs
        sys_msg = None
        conv = []
        # 分离系统提示和普通对话
        for m in msgs:
            if m['role'] == 'system' and keep_sys:
                sys_msg = m
            else:
                conv.append(m)

        # 计算可用token
        avail = max_tok - config['reserve_token']
        if sys_msg:
            avail -= self.get_msgs_tokens([sys_msg])

        # 从后往前保留对话
        res = []
        now = 0
        for m in reversed(conv):
            t = self.get_msgs_tokens([m])
            if now + t > avail:
                break
            res.insert(0, m)
            now += t

        # 拼接系统提示和截断后的对话
        final = []
        if sys_msg:
            final.append(sys_msg)
        final.extend(res)
        return final

# 模型基类：定义统一接口
class BaseModel:
    def __init__(self, api, mid, mname, ctx):
        self.api = api          # API源标识
        self.mid = mid          # 模型ID
        self.mname = mname      # 模型名称
        self.ctx = ctx          # 上下文窗口大小
        self.tk = TokenManager(mname)  # Token管理器
    async def chat(self, msgs):
        pass  # 子类需实现此方法

# OpenAI兼容模型适配器（DeepSeek、通义千问、OpenAI）
class OpenAIAdapter(BaseModel):
    def __init__(self, api, mid, mname, ctx):
        super().__init__(api, mid, mname, ctx)
        cfg = api_sources[api]
        # 初始化异步OpenAI客户端
        self.client = AsyncOpenAI(api_key=cfg['key'], base_url=cfg['url'])
    async def chat(self, msgs):
        # 截断历史
        m = self.tk.cut_history(msgs, self.ctx)
        # 调用API
        resp = await self.client.chat.completions.create(model=self.mname,messages=m,temperature=0.7)
        return resp.choices[0].message.content.strip()

# Gemini模型适配器
class GeminiAdapter(BaseModel):
    def __init__(self, api, mid, mname, ctx):
        super().__init__(api, mid, mname, ctx)
        genai.configure(api_key=api_sources[api]['key'])
        self.client = genai.GenerativeModel(mname)
    # 转换消息格式为Gemini要求的格式
    def convert(self, msgs):
        r = []
        for m in msgs:
            role = 'user' if m['role'] == 'user' else 'model'
            r.append({'role':role,'parts':[m['content']]})
        return r
    async def chat(self, msgs):
        m = self.tk.cut_history(msgs, self.ctx)
        gmsg = self.convert(m)
        resp = await self.client.generate_content_async(gmsg)
        return resp.text.strip()

# Claude模型适配器
class ClaudeAdapter(BaseModel):
    def __init__(self, api, mid, mname, ctx):
        super().__init__(api, mid, mname, ctx)
        self.client = AsyncAnthropic(api_key=api_sources[api]['key'])
    async def chat(self, msgs):
        m = self.tk.cut_history(msgs, self.ctx)
        sys = ''
        conv = []
        # 分离系统提示
        for i in m:
            if i['role'] == 'system':
                sys = i['content']
            else:
                conv.append(i)
        # 调用Claude API
        resp = await self.client.messages.create(model=self.mname,max_tokens=config['reserve_token'],system=sys,messages=conv)
        return resp.content[0].text.strip()

# 注册所有可用模型
def reg_models():
    r = []
    # DeepSeek模型
    r.append(OpenAIAdapter('deepseek','deepseek-chat','deepseek-chat',128000))
    r.append(OpenAIAdapter('deepseek','deepseek-coder-v2','deepseek-coder-v2',128000))
    # 通义千问模型
    r.append(OpenAIAdapter('qwen','qwen-turbo','qwen-turbo',128000))
    r.append(OpenAIAdapter('qwen','qwen-plus','qwen-plus',128000))
    r.append(OpenAIAdapter('qwen','qwen-max','qwen-max',200000))
    r.append(OpenAIAdapter('qwen','qwen-coder','qwen-coder',128000))
    # OpenAI模型
    r.append(OpenAIAdapter('openai','gpt-3.5-turbo','gpt-3.5-turbo',16384))
    r.append(OpenAIAdapter('openai','gpt-4o','gpt-4o',128000))
    # Gemini模型
    r.append(GeminiAdapter('gemini','gemini-1.5-flash','gemini-1.5-flash',1048576))
    r.append(GeminiAdapter('gemini','gemini-1.5-pro','gemini-1.5-pro',1048576))
    # Claude模型
    r.append(ClaudeAdapter('claude','claude-3-5-sonnet','claude-3-5-sonnet-20240620',200000))
    r.append(ClaudeAdapter('claude','claude-3-opus','claude-3-opus-20240229',200000))
    return r
model_reg = reg_models()  # 初始化模型注册表

# 对话历史管理类
class History:
    def __init__(self, path):
        self.path = path  # 历史文件路径
        self.msgs = []    # 对话消息列表
        self.load()       # 加载历史
    # 从文件加载历史
    def load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path,'r',encoding='utf8') as f:
                    self.msgs = json.load(f)
            except:
                self.msgs = []  # 加载失败则清空
    # 保存历史到文件
    def save(self):
        try:
            with open(self.path,'w',encoding='utf8') as f:
                json.dump(self.msgs,f,ensure_ascii=False,indent=2)
        except:
            pass  # 保存失败则忽略
    # 添加一条消息
    def add(self, role, txt):
        self.msgs.append({'role':role,'content':txt})
        self.save()
    # 重置历史
    def reset(self):
        self.msgs = []
        if os.path.exists(self.path):
            os.remove(self.path)
        print_success('对话历史已清空')
    # 获取格式化的历史记录
    def show(self, n=10):
        if not self.msgs:
            return '无对话历史'
        d = self.msgs[-n*2:]  # 取最近n*2条消息
        t = ''
        for i,m in enumerate(d):
            r = '用户' if m['role']=='user' else 'AI'
            # 超长内容截断显示
            t += f'[{i+1}] {r}: {m["content"][:100]}...\n' if len(m['content'])>100 else f'[{i+1}] {r}: {m["content"]}\n'
        return t.strip()

# 导入打印工具
from prompt_toolkit import print_formatted_text
# 打印分割线
def print_border():
    print_formatted_text('─' * 60, style=cli_style)
# 打印带边框的标题
def print_title(txt):
    print_border()
    print_formatted_text(f' {txt} ', style=cli_style)
    print_border()
# 打印系统信息
def print_sys(txt):
    print_formatted_text(txt, style=cli_style)
# 打印错误信息
def print_err(txt):
    print_formatted_text(f'❌ {txt}', style=cli_style)
# 打印成功信息
def print_success(txt):
    print_formatted_text(f'✅ {txt}', style=cli_style)

# 代码高亮函数
def high_code(txt):
    # 匹配Markdown代码块
    p = re.compile(r'```(\w*)\n(.*?)\n```', re.DOTALL)
    def rep(m):
        lang = m.group(1)
        c = m.group(2)
        try:
            # 根据语言选择词法分析器，或自动识别
            lex = get_lexer_by_name(lang) if lang else guess_lexer(c)
            return highlight(c, lex, TerminalFormatter())
        except:
            return f'```\n{c}\n```'  # 高亮失败则返回原代码块
    return p.sub(rep, txt)

# 解析@文件引用
def read_file_ref(txt):
    # 匹配@文件名格式
    p = re.compile(r'@(\S+\.\w+)')
    def f(m):
        path = m.group(1)
        if not os.path.exists(path):
            print_err(f'文件不存在 {path}')
            return m.group(0)
        try:
            with open(path,encoding='utf8') as f:
                c = f.read()
            return f'\n📄 {path} \n```\n{c}\n```\n'  # 返回文件内容
        except:
            print_err('文件读取失败')
            return m.group(0)
    return p.sub(f, txt)

# 命令处理器类
class Cmd:
    def __init__(self, hist, reg, api, model):
        self.hist = hist          # 历史管理器
        self.reg = reg            # 模型注册表
        self.now_api = api        # 当前API源
        self.now_model = model    # 当前模型
        # 命令映射
        self.cmds = {
            '/exit':self.exit,
            '/help':self.help,
            '/clear':self.clear,
            '/reset':self.reset,
            '/history':self.history,
            '/model':self.model,
            '/api':self.api
        }
    # 判断是否为命令
    def is_cmd(self, t):
        return t.strip().startswith('/')
    # 执行命令
    async def run(self, t):
        ps = t.strip().split()
        cmd = ps[0].lower()
        args = ps[1:]
        if cmd not in self.cmds:
            print_err(f'未知命令: {cmd}')
            return False
        return await self.cmds[cmd](sslocal://flow/file_open?url=args&flow_extra=eyJsaW5rX3R5cGUiOiJjb2RlX2ludGVycHJldGVyIn0=)
    # 退出命令
    async def exit(self, args):
        print_success('感谢')
        return True
    # 帮助命令
    async def help(self, args):
        print_title('帮助')
        t = '''
/help        查看帮助
/exit        退出程序
/clear       清空屏幕
/reset       重置对话
/history     查看历史
/model list  查看模型 | /model switch [名称]
/api list    查看API   | /api switch [名称] | /api info
'''
        print_sys(t)
        return False
    # 清屏命令
    async def clear(self, args):
        os.system('cls' if os.name=='nt' else 'clear')
        return False
    # 重置历史命令
    async def reset(self, args):
        self.hist.reset()
        return False
    # 查看历史命令
    async def history(self, args):
        print_title('历史对话')
        n = int(args[0]) if args and args[0].isdigit() else 10
        print_sys(self.hist.show(n))
        return False
    # 模型管理命令
    async def model(self, args):
        if not args:
            print_err('用法: /model list /switch')
            return False
        s = args[0].lower()
        if s == 'list':
            print_title('可用模型')
            t = ''
            for k,v in api_sources.items():
                t+=f'\n【{v["name"]}】\n'
                for m in self.reg:
                    if m.api == k:
                        cur = '👉 当前' if m.mid==self.now_model.mid else ''
                        t+=f'  {m.mid} {cur}\n'
            print_sys(t)
        elif s == 'switch':
            if len(args)<2:
                print_err('输入模型ID')
                return False
            target = args[1]
            m = None
            for i in self.reg:
                if i.mid == target:
                    m=i
                    break
            if not m:
                print_err('模型不存在')
                return False
            self.now_model = m
            self.now_api = m.api
            print_success(f'已切换模型: {m.mid}')
        return False
    # API管理命令
    async def api(self, args):
        if not args:
            print_err('用法: /api list /switch /info')
            return False
        s = args[0].lower()
        if s == 'list':
            print_title('API源列表')
            t = ''
            for k,v in api_sources.items():
                ok = '已配置' if v['key'] else '未配置'
                cur = ' 当前' if k==self.now_api else ''
                t+=f'{k} | {v["name"]} | {ok} {cur}\n'
            print_sys(t)
        elif s == 'switch':
            if len(args)<2:
                print_err('请输入API名称')
                return False
            target = args[1].lower()
            if target not in api_sources:
                print_err('API不存在')
                return False
            if not api_sources[target]['key']:
                print_err('API密钥未配置')
                return False
            self.now_api = target
            dmodel = api_sources[target]['def_model']
            for m in self.reg:
                if m.mid == dmodel and m.api == target:
                    self.now_model = m
                    break
            print_success(f'切换API: {target}')
            print_success(f'模型: {dmodel}')
        elif s == 'info':
            print_title('当前信息')
            t = f'API: {api_sources[self.now_api]["name"]}\n模型: {self.now_model.mid}\n'
            print_sys(t)
        return False

# 主函数
async def main():
    # 初始化历史管理器
    hist = History(config['hist_file'])
    now_api = config['default_api']
    # 校验默认API
    if now_api not in api_sources:
        now_api = 'deepseek'
    # 初始化默认模型
    now_model = None
    dmid = api_sources[now_api]['def_model']
    for m in model_reg:
        if m.mid == dmid and m.api == now_api:
            now_model = m
            break
    # 兜底：使用第一个模型
    if not now_model:
        now_model = model_reg[0]
        now_api = now_model.api

    # 初始化命令处理器
    cmd = Cmd(hist, model_reg, now_api, now_model)
    # 初始化交互式会话
    session = PromptSession(history=FileHistory(config['input_hist']),auto_suggest=AutoSuggestFromHistory())

    # 清屏并显示欢迎界面
    os.system('cls' if os.name=='nt' else 'clear')
    print_title('交互式命令行工具')
    welcome = '''
支持 DeepSeek | 通义千问
输入 /help 查看命令 | 输入 @文件名 读取文件
'''
    print_sys(welcome)
    print_success(f'当前API: {api_sources[now_api]["name"]}')
    print_success(f'当前模型: {now_model.mid}')
    print_border()

    # 主循环
    while 1:
        try:
            # 获取用户输入
            user = await session.prompt_async(HTML('<user-prompt>👤 You > </user-prompt>'),style=cli_style)
            user = user.strip()
            if not user:
                continue
            # 处理命令
            if cmd.is_cmd(user):
                ex = await cmd.run(user)
                if ex:
                    break
                now_api = cmd.now_api
                now_model = cmd.now_model
                continue

            # 解析文件引用
            user = read_file_ref(user)
            hist.add('user', user)

            # 调用AI并显示回复
            print()
            print_formatted_text(f'AI ({api_sources[now_api]["name"]}) > ', style=cli_style)
            print_border()
            res = await now_model.chat(hist.msgs)
            print(high_code(res))  # 代码高亮
            print_border()
            print()
            hist.add('assistant', res)

        # Ctrl+C中断
        except KeyboardInterrupt:
            print_sys('\n输入 /exit 退出')
            continue
        # Ctrl+D退出
        except EOFError:
            print_success('再见！')
            break
        # 其他异常
        except Exception as e:
            print_err(f'运行错误')

# 程序入口
if __name__ == '__main__':
    asyncio.run(main())
