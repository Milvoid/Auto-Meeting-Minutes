import os
import re
import openai
from pathlib import Path

class MeetingNotesComparer:
    def __init__(self, api_key, base_url, model, work_dir, file_names, meeting_comparer_system_prompt, meeting_comparer_note_prompt):
        self.api_key = api_key
        self.base_url = base_url
        self.work_dir = Path(work_dir)
        self.file_names = file_names
        self.file_paths = [self.work_dir / file_name for file_name in file_names]
        self.meeting_comparer_system_prompt = meeting_comparer_system_prompt
        self.meeting_comparer_note_prompt = meeting_comparer_note_prompt
        self.model = model
        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)

    def read_and_split_files(self):
        """ 读取所有文件并按 '---' 进行分割 """
        file_contents = []
        for path in self.file_paths:
            with open(path, "r", encoding="utf-8") as f:
                file_contents.append(f.read().strip().split("\n---\n"))
        
        # 确保所有文件的区块数量一致
        block_counts = [len(content) for content in file_contents]
        if len(set(block_counts)) > 1:
            raise ValueError(f"错误: 文件区块数量不匹配: {block_counts}")
        
        return list(zip(*file_contents))  # 转置，按区块分组

    def format_blocks(self, block_group):
        """ 格式化区块内容 """
        formatted_text = ""
        for i, content in enumerate(block_group):
            file_name_without_ext = os.path.splitext(self.file_names[i])[0]  # 去除文件后缀
            formatted_text += f"[{file_name_without_ext} 转写]\n{content.strip()}\n\n"
        return formatted_text.strip()

    def call_openai_api(self, block_text):
        """ 调用 LLM 进行优化 """
        messages = [
            {"role": "system", "content": self.meeting_comparer_system_prompt},
            {"role": "user", "content": f"关于这份文本的备注：\n\n{self.meeting_comparer_note_prompt}\n\n你需要检查的文本内容：\n\n{block_text}"}
        ]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7
        )

        print(f"\n 此区块输出\n\n：{response.choices[0].message.content}\n")
        
        return response.choices[0].message.content.strip()

    def process(self, output_filename = "Checked-Transcription.md"):
        """ 主处理逻辑 """
        block_groups = self.read_and_split_files()
        results = []
        
        for i, block_group in enumerate(block_groups):
            print(f"正在处理第 {i+1} 个区块...")
            formatted_text = self.format_blocks(block_group)
            optimized_text = self.call_openai_api(formatted_text)
            results.append(f"\n\n{optimized_text}")
        
        output_path = self.work_dir / output_filename
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(results))
        
        print(f"优化后的会议纪要已保存至 {output_path}")

        return results


class MeetingOutlineGenerator:
    def __init__(self, api_key, base_url, model, work_dir, input_filename, outline_prompt):
        self.api_key = api_key
        self.base_url = base_url
        self.work_dir = Path(work_dir)
        self.input_filename = input_filename
        self.input_filepath = self.work_dir / input_filename
        self.outline_prompt = outline_prompt
        self.model = model
        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)

    def read_input_file(self):
        """ 读取文字稿 """
        with open(self.input_filepath, "r", encoding="utf-8") as f:
            return f.read().strip()

    def call_openai_api(self, text):
        """ 调用 LLM 生成大纲 """
        messages = [
            {"role": "system", "content": self.outline_prompt},
            {"role": "user", "content": f"会议文字稿：\n\n{text}"}
        ]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()

    def process(self, output_filename):
        """ 生成会议大纲 """
        text = self.read_input_file()
        outline = self.call_openai_api(text)
        
        output_path = self.work_dir / output_filename
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(outline)
        
        print(f"会议纪要大纲已保存至 {output_path}")
        
        return outline



class MeetingNotesWriter:
    def __init__(self, api_key, base_url, model, work_dir, transcript_filename, outline_filename, writing_prompt):
        self.api_key = api_key
        self.base_url = base_url
        self.work_dir = Path(work_dir)
        self.transcript_filepath = self.work_dir / transcript_filename
        self.outline_filepath = self.work_dir / outline_filename
        self.writing_prompt = writing_prompt
        self.model = model
        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)

    def read_file(self, filepath):
        """ 读取文件内容 """
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read().strip()

    def split_outline_into_parts(self, outline_text):
        """ 解析大纲，按 <part [num]> 标签拆分 """
        parts = re.findall(r'<part (\d+)>(.*?)</part>', outline_text, re.DOTALL)
        return {int(num): content.strip() for num, content in parts}

    def call_openai_api(self, transcript, part_outline):
        """ 调用 LLM 生成完整纪要 """
        messages = [
            {"role": "system", "content": self.writing_prompt},
            {"role": "user", "content": f"完整会议文字稿：\n\n{transcript}\n\n对应模块大纲：\n\n{part_outline}"}
        ]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()

    def process(self, output_filename):
        """ 处理并生成完整会议纪要 """
        transcript_text = self.read_file(self.transcript_filepath)
        outline_text = self.read_file(self.outline_filepath)
        outline_parts = self.split_outline_into_parts(outline_text)
        
        results = []
        for part_num, part_outline in outline_parts.items():
            print(f"正在处理第 {part_num} 部分...")
            part_summary = self.call_openai_api(transcript_text, part_outline)
            results.append(f"<part {part_num}>\n{part_summary}\n</part>")
        
        output_path = self.work_dir / output_filename
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(results))
        
        print(f"完整会议纪要已保存至 {output_path}")
        
        return "\n\n".join(results)

class MeetingNotesChecker:
    def __init__(self, api_key, base_url, model, work_dir, transcript_filename, notes_filename, checking_prompt):
        self.api_key = api_key
        self.base_url = base_url
        self.work_dir = Path(work_dir)
        self.transcript_filepath = self.work_dir / transcript_filename
        self.notes_filepath = self.work_dir / notes_filename
        self.checking_prompt = checking_prompt
        self.model = model
        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)

    def read_file(self, filepath):
        """ 读取文件内容 """
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read().strip()

    def split_notes_into_parts(self, notes_text):
        """ 解析会议纪要，按 <part [num]> 标签拆分 """
        parts = re.findall(r'<part (\d+)>\n(.*?)\n</part>', notes_text, re.DOTALL)
        return {int(num): content.strip() for num, content in parts}

    def call_openai_api(self, transcript, part_notes):
        """ 调用 OpenAI 进行事实性检查 """
        messages = [
            {"role": "system", "content": self.checking_prompt},
            {"role": "user", "content": f"完整会议文字稿：\n\n{transcript}\n\n对应模块纪要：\n\n{part_notes}"}
        ]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()

    def process(self, output_filename):
        """ 处理并检查会议纪要的事实性错误 """
        transcript_text = self.read_file(self.transcript_filepath)
        notes_text = self.read_file(self.notes_filepath)
        notes_parts = self.split_notes_into_parts(notes_text)
        
        results = []
        for part_num, part_notes in notes_parts.items():
            print(f"正在检查第 {part_num} 部分...")
            checked_summary = self.call_openai_api(transcript_text, part_notes)
            results.append(f"<part {part_num}>\n{checked_summary}\n</part>")
        
        output_path = self.work_dir / output_filename
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(results))
        
        print(f"检查后的会议纪要已保存至 {output_path}")
        
        return "\n\n".join(results)
    

if __name__ == "__main__":

    # Working Directory
    WORK_DIR = ""

    # API Configuration
    API_KEY = ""
    BASE_URL = ""
    MODEL = ""

    # Meeting Transcript Check Configuration
    FILE_NAMES = ["Whisper.txt", "Tongyi.txt"]

    MEETING_COMPARER_SYSTEM_PROMPT = '''
    你现在需要完成交叉核对用户给出的语音转文字稿的任务。对于用户提供的来自不同软件转录的同一个音频的文字稿，你需要对它们完成复核，最后给出一份正确的文字稿，并按要求进行备注。

    具体要求：
    - 保证正确的自然段换行，**不要每句话一行**
    - 两份转录稿一致、且符合语境的地方，直接输出
    - 两份转录稿有不一致的地方，如果有一方为明显错误，请你保留正确的一方，并将不正确的一方用括号和红色字体在后面注明，例如 （<font color="red">[明显错误：文本来源] 原文</font>）
    - 两份转录稿有不一致的地方，但是无法判断哪一方错误，请用 <font color="orange">[对应文本来源名称1: 对应原文1| 对应文本来源名称2: 对应原文2]</font> 这样的写法标出供人工判断
    - **任何数字**的地方均用  <font color="green">数字</font> 来注明以供人工复核。

    注意：你的任务只是交叉比对并复核文字稿，不需要对文字稿进行任何额外加工，最终应该输出最接近真实录音的文字稿，并附带上面要求的标注。
    '''

    MEETING_COMPARER_NOTE_PROMPT = '''
    本会议的主题关于
    '''

    # Outline Generate Configuration
    OUTLINE_PROMPT = '''
    请根据用户提供的会议文字稿撰写一份清晰、精炼、逻辑严密的**投资会议纪要大纲**，使基金经理及不熟悉该公司的投资者能够快速理解公司现状、行业竞争格局及未来增长点。请遵循以下要求：

    ### **纪要撰写目标**
    1. **让不懂的人看懂**：假设读者不熟悉该公司或行业，需要进行基本科普，解释关键术语，并确保信息易于理解。
    2. **高效传递信息**：
    - 让读者**在短时间内抓住核心信息**，先呈现关键结论，再提供支撑论据。
    - **避免流水账**，去除无意义的对话和重复表述，直接提炼结论。
    - **语言简洁**，减少冗余表述，尽量用精准的行业数据和事实支持观点。
    3. **逻辑清晰**：
    - **合理调整内容顺序**，确保信息流畅且符合逻辑。

    ### **具体撰写要求**
    1. **用小标题、加粗、数据支持信息**
    - 例：“**2024年销量增长5%-7%，市占率持续提升**” 而不是 “公司销量增长较快”。
    2. **减少主观判断，强调数据和事实**
    - 例：“**公司2024年Q2因价格战阶段性让利11个百分点，导致毛利率承压**”，而不是“公司毛利率下降，可能与价格战有关”。
    3. **避免记录无意义的对话**
    - 直接提炼关键信息，而不是逐字转录讲话内容。
    4. **补充必要背景信息，填补逻辑漏洞**
    - 例如专家只提到“Q1客户需求一般，Q2预计转好”，需要补充：
        - 具体哪些客户需求一般？
        - Q2的回暖原因是什么？
        - 相关数据或市场趋势是什么？

    ### **内容要求**
    1. 必须**严格按照参考来源**生成，重点是符合原稿，不要编造或使用任何文中没有提到的内容
    2. 大纲撰写时内容必须**全面覆盖原稿内容**，不要遗漏
        
    ## [IMPORTANT] 格式要求

    你需要对你的大纲中模块的划分**必须**使用  `<part n>\n 第n部分大纲内容 \n</part>` 这样的标签来注明，以便程序处理（所有标签外的内容都会被忽略，仅标签内视为有效）。
    例如 `<part 4>\n 这是一块第四部分大纲内容 \n</part> 这是另一块第四部分大纲内容` 这样的文本中，只有标签内的部分会被匹配到，不在标签里的部分视为无效。

    请基于上述要求，对以下会议文字稿进行总结和优化，撰写一份结构清晰、重点突出的**投资会议纪要**。
    '''


    # Writing Configuration
    WRITING_PROMPT = '''
    请根据用户提供的会议文字稿以及会议纪要大纲的对应部分，撰写这部分清晰、精炼、逻辑严密的**投资会议纪要**，使基金经理及不熟悉该公司的投资者能够快速理解公司现状、行业竞争格局及未来增长点。请遵循以下要求：

    ### **纪要撰写目标**
    1. **让不懂的人看懂**：假设读者不熟悉该公司或行业，需要进行基本科普，解释关键术语，并确保信息易于理解。
    2. **高效传递信息**：
    - 让读者**在短时间内抓住核心信息**，先呈现关键结论，再提供支撑论据。
    - **避免流水账**，去除无意义的对话和重复表述，直接提炼结论。
    - **语言简洁**，减少冗余表述，尽量用精准的行业数据和事实支持观点。
    3. **逻辑清晰**：
    - **合理调整内容顺序**，确保信息流畅且符合逻辑。

    ### **具体撰写要求**
    1. **用小标题、加粗、数据支持信息**
    - 例：“**2024年销量增长5%-7%，市占率持续提升**” 而不是 “公司销量增长较快”。
    2. **减少主观判断，强调数据和事实**
    - 例：“**公司2024年Q2因价格战阶段性让利11个百分点，导致毛利率承压**”，而不是“公司毛利率下降，可能与价格战有关”。
    3. **避免记录无意义的对话**
    - 直接提炼关键信息，而不是逐字转录讲话内容。
    4. **补充必要背景信息，填补逻辑漏洞**
    - 例如专家只提到“Q1客户需求一般，Q2预计转好”，需要补充：
        - 具体哪些客户需求一般？
        - Q2的回暖原因是什么？
        - 相关数据或市场趋势是什么？

    ### **内容要求**
    1. 必须**严格按照参考来源**生成，重点是符合原稿，不要编造或使用任何文中没有提到的内容
    2. 撰写时内容必须**全面覆盖原稿内容**，不要遗漏
    3. 每一点都至少要用两三句话，一两百字讲清楚；并且必须符合原稿，不得编造

    请基于上述要求，根据以下会议文字稿，以及给定的部分的大纲，撰写这部分结构清晰、重点突出的**投资会议纪要**。
    '''

    # Checking Prompt
    CHECKING_PROMPT = '''
    你现在需要**逐条核实**会议纪要中是否出现了会议原文中没有的信息。注意，核对的标准并不是纪要内容是否正确或者合理，而是**是否在原稿中有直接说明**。
    每一句话都检查，并注明 "✅ 原文明确提及" "⚠️ 原文部分提及" "❌ 原文未提及"。
    如果确认原文有出现或部分出现，用 > 的引用符号 Quote 出原句。
    每句话检查后都给出简短的检查结果判断理由。
    记住不要泛泛地检查，而是需要一点一点、一句一句地详细检查。
    '''

    # 初始化并运行 MeetingNotesComparer
    comparer = MeetingNotesComparer(API_KEY, BASE_URL, MODEL, WORK_DIR, FILE_NAMES, MEETING_COMPARER_SYSTEM_PROMPT, MEETING_COMPARER_NOTE_PROMPT)
    compared_results = comparer.process("Checked-Transcription.md")
    
    # 初始化并运行 MeetingOutlineGenerator
    outline_generator = MeetingOutlineGenerator(API_KEY, BASE_URL, MODEL, WORK_DIR, "Checked-Transcription.md", OUTLINE_PROMPT)
    outline_results = outline_generator.process("Meeting-Outline.md")
    
    # 初始化并运行 MeetingNotesWriter
    notes_writer = MeetingNotesWriter(API_KEY, BASE_URL, MODEL, WORK_DIR, "Checked-Transcription.md", "Meeting-Outline.md", WRITING_PROMPT)
    written_notes = notes_writer.process("Meeting-Notes.md")
    
    # 初始化并运行 MeetingNotesChecker
    notes_checker = MeetingNotesChecker(API_KEY, BASE_URL, MODEL, WORK_DIR, "Checked-Transcription.md", "Meeting-Notes.md", CHECKING_PROMPT)
    checked_notes = notes_checker.process("Checked-Meeting-Notes.md")