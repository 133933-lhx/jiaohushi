import os
import json
import re
import asyncio
from dotenv import load_dotenv
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.styles import Style
from pygments import highlight
from pygments.lexers import get_lexer_by_name, guess_lexer
from pygments.formatters import TerminalFormatter
import tiktoken

from openai import AsyncOpenAI
try:
    import google.genai as genai
except:

    import google.generativeai as genai
from anthropic import AsyncAnthropic

load_dotenv()
config = {
    'ctx_win': int(os.getenv('DEFAULT_CONTEXT_WINDOW', 8192)),
    'hist_file': 'chat_history.json',
    'input_hist': '.input_history',
    'reserve_token': 1024,
    'default_api': os.getenv('DEFAULT_API_SOURCE', 'deepseek'),
}
cli_style = Style.from_dict({
    'user-prompt': '#66ff66 bold',
    'ai-prompt': '#3399ff bold',
    'system': '#ffcc00 italic',
    'error': '#ff4444 bold',
    'success': '#00ff99 bold',
})

api_sources = {
    'deepseek': {
        'name': 'DeepSeek',
        'key': os.getenv('DEEPSEEK_API_KEY'),
        'url': 'https://api.deepseek.com/v1',
        'def_model': 'deepseek-chat',
        'models': ['deepseek-chat', 'deepseek-coder-v2']
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
        'url': '',
        'def_model': 'gemini-1.5-flash',
        'models': ['gemini-1.5-flash', 'gemini-1.5-pro']
    },
    'claude': {
        'name': 'Anthropic Claude',
        'key': os.getenv('ANTHROPIC_API_KEY'),
        'url': '',
        'def_model': 'claude-3-5-sonnet',
        'models': ['claude-3-5-sonnet', 'claude-3-opus']
    }
}

class TokenManager:
    def __init__(self, model='gpt-3.5-turbo'):
        self.encoder = tiktoken.get_encoding('cl100k_base')
        self.model = model

    def get_tokens(self, text):
        return len(self.encoder.encode(text))

    def get_msgs_tokens(self, msgs):
        total = 0
        for m in msgs:
            total += self.get_tokens(m['content'])
            total += self.get_tokens(m['role'])
        return total

    def cut_history(self, msgs, max_tok, keep_sys=True):
        if not msgs:
            return msgs
        sys_msg = None
        conv = []
        for m in msgs:
            if m['role'] == 'system' and keep_sys:
                sys_msg = m
            else:
                conv.append(m)

        avail = max_tok - config['reserve_token']
        if sys_msg:
            avail -= self.get_msgs_tokens([sys_msg])

        res = []
        now = 0
    
        for m in reversed(conv):
            t = self.get_msgs_tokens([m])
            if now + t > avail:
                break
            res.insert(0, m)
            now += t

        final = []
        if sys_msg:
            final.append(sys_msg)
        final.extend(res)
        return final

class BaseModel:
    def __init__(self, api, mid, mname, ctx):
        self.api = api
        self.mid = mid
        self.mname = mname
        self.ctx = ctx
        self.tk = TokenManager(mname)
    async def chat(self, msgs):
        pass

class OpenAIAdapter(BaseModel):
    def __init__(self, api, mid, mname, ctx):
        super().__init__(api, mid, mname, ctx)
        cfg = api_sources[api]
        self.client = AsyncOpenAI(api_key=cfg['key'], base_url=cfg['url'])
    async def chat(self, msgs):
        m = self.tk.cut_history(msgs, self.ctx)
        resp = await self.client.chat.completions.create(model=self.mname,messages=m,temperature=0.7)
        return resp.choices[0].message.content.strip()

class GeminiAdapter(BaseModel):
    def __init__(self, api, mid, mname, ctx):
        super().__init__(api, mid, mname, ctx)
        genai.configure(api_key=api_sources[api]['key'])
        self.client = genai.GenerativeModel(mname)
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

class ClaudeAdapter(BaseModel):
    def __init__(self, api, mid, mname, ctx):
        super().__init__(api, mid, mname, ctx)
        self.client = AsyncAnthropic(api_key=api_sources[api]['key'])
    async def chat(self, msgs):
        m = self.tk.cut_history(msgs, self.ctx)
        sys = ''
        conv = []
        for i in m:
            if i['role'] == 'system':
                sys = i['content']
            else:
                conv.append(i)
        resp = await self.client.messages.create(model=self.mname,max_tokens=config['reserve_token'],system=sys,messages=conv)
        return resp.content[0].text.strip()

def reg_models():
    r = []
    r.append(OpenAIAdapter('deepseek','deepseek-chat','deepseek-chat',128000))
    r.append(OpenAIAdapter('deepseek','deepseek-coder-v2','deepseek-coder-v2',128000))
    r.append(OpenAIAdapter('qwen','qwen-turbo','qwen-turbo',128000))
    r.append(OpenAIAdapter('qwen','qwen-plus','qwen-plus',128000))
    r.append(OpenAIAdapter('qwen','qwen-max','qwen-max',200000))
    r.append(OpenAIAdapter('qwen','qwen-coder','qwen-coder',128000))
    r.append(OpenAIAdapter('openai','gpt-3.5-turbo','gpt-3.5-turbo',16384))
    r.append(OpenAIAdapter('openai','gpt-4o','gpt-4o',128000))
    r.append(GeminiAdapter('gemini','gemini-1.5-flash','gemini-1.5-flash',1048576))
    r.append(GeminiAdapter('gemini','gemini-1.5-pro','gemini-1.5-pro',1048576))
    r.append(ClaudeAdapter('claude','claude-3-5-sonnet','claude-3-5-sonnet-20240620',200000))
    r.append(ClaudeAdapter('claude','claude-3-opus','claude-3-opus-20240229',200000))
    return r
model_reg = reg_models()

class History:
    def __init__(self, path):
        self.path = path
        self.msgs = []
        self.load()
    def load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path,'r',encoding='utf8') as f:
                    self.msgs = json.load(f)
            except:
               
                self.msgs = []
    def save(self):
        try:
            with open(self.path,'w',encoding='utf8') as f:
                json.dump(self.msgs,f,ensure_ascii=False,indent=2)
        except:
           
            pass
    def add(self, role, txt):
        self.msgs.append({'role':role,'content':txt})
        self.save()
    def reset(self):
        self.msgs = []
        if os.path.exists(self.path):
            os.remove(self.path)
        print_success('对话历史已清空')
    def show(self, n=10):
        if not self.msgs:
            return '无对话历史'
        d = self.msgs[-n*2:]
        t = ''
        for i,m in enumerate(d):
            r = '用户' if m['role']=='user' else 'AI'
            
            t += f'[{i+1}] {r}: {m["content"][:100]}...\n' if len(m['content'])>100 else f'[{i+1}] {r}: {m["content"]}\n'
        return t.strip()

from prompt_toolkit import print_formatted_text
def print_border():
    print_formatted_text('─' * 60, style=cli_style)
def print_title(txt):
    print_border()
    print_formatted_text(f' {txt} ', style=cli_style)
    print_border()
def print_sys(txt):
    print_formatted_text(txt, style=cli_style)
def print_err(txt):
    print_formatted_text(f'❌ {txt}', style=cli_style)
def print_success(txt):
    print_formatted_text(f'✅ {txt}', style=cli_style)

def high_code(txt):
    p = re.compile(r'```(\w*)\n(.*?)\n```', re.DOTALL)
    def rep(m):
        lang = m.group(1)
        c = m.group(2)
        try:
            lex = get_lexer_by_name(lang) if lang else guess_lexer(c)
            return highlight(c, lex, TerminalFormatter())
        except:
            return f'```\n{c}\n```'
    return p.sub(rep, txt)

def read_file_ref(txt):
    p = re.compile(r'@(\S+\.\w+)')
    def f(m):
        path = m.group(1)
        if not os.path.exists(path):
            print_err(f'文件不存在 {path}')
            return m.group(0)
        try:
            with open(path,encoding='utf8') as f:
                c = f.read()
            return f'\n📄 {path} \n```\n{c}\n```\n'
        except:
            print_err('文件读取失败')
            return m.group(0)
    return p.sub(f, txt)

class Cmd:
    def __init__(self, hist, reg, api, model):
        self.hist = hist
        self.reg = reg
        self.now_api = api
        self.now_model = model
        self.cmds = {
            '/exit':self.exit,
            '/help':self.help,
            '/clear':self.clear,
            '/reset':self.reset,
            '/history':self.history,
            '/model':self.model,
            '/api':self.api
        }
    def is_cmd(self, t):
        return t.strip().startswith('/')
    async def run(self, t):
        ps = t.strip().split()
        cmd = ps[0].lower()
        args = ps[1:]
        if cmd not in self.cmds:
            print_err(f'未知命令: {cmd}')
            return False
        return await self.cmds[cmd](args)
    async def exit(self, args):
        print_success('感谢')
        return True
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
    async def clear(self, args):
        os.system('cls' if os.name=='nt' else 'clear')
        return False
    async def reset(self, args):
        self.hist.reset()
        return False
    async def history(self, args):
        print_title('历史对话')
        n = int(args[0]) if args and args[0].isdigit() else 10
        print_sys(self.hist.show(n))
        return False
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
async def main():
    hist = History(config['hist_file'])
    now_api = config['default_api']
    if now_api not in api_sources:
        now_api = 'deepseek'
    now_model = None
    dmid = api_sources[now_api]['def_model']
    for m in model_reg:
        if m.mid == dmid and m.api == now_api:
            now_model = m
            break
    if not now_model:
        now_model = model_reg[0]
        now_api = now_model.api

    cmd = Cmd(hist, model_reg, now_api, now_model)
    session = PromptSession(history=FileHistory(config['input_hist']),auto_suggest=AutoSuggestFromHistory())

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

    while 1:
        try:
            user = await session.prompt_async(HTML('<user-prompt>👤 You > </user-prompt>'),style=cli_style)
            user = user.strip()
            if not user:
                continue
            if cmd.is_cmd(user):
                ex = await cmd.run(user)
                if ex:
                    break
                now_api = cmd.now_api
                now_model = cmd.now_model
                continue

            user = read_file_ref(user)
            hist.add('user', user)

          
            print()
            print_formatted_text(f'AI ({api_sources[now_api]["name"]}) > ', style=cli_style)
            print_border()
            res = await now_model.chat(hist.msgs)
        
            print(high_code(res))
            print_border()
            print()
            hist.add('assistant', res)

       
        except KeyboardInterrupt:
            print_sys('\n输入 /exit 退出')
            continue
        except EOFError:
            print_success('再见！')
            break
       
        except Exception as e:
      
            print_err(f'运行错误')

if __name__ == '__main__':
    asyncio.run(main())
