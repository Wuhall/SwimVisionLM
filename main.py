import os
import cv2
import base64
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr
from datetime import datetime
import shutil

class SwimAnalysisTool:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key)
        self.output_dir = "output_frames"
        self.clean_output_dir()

    def clean_output_dir(self):
        """Clean up the output directory."""
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir)

    def extract_frames(self, video_path):
        """Extract frames from video."""
        cap = cv2.VideoCapture(video_path)
        base64_frames = []
        sampled_images = []
        frame_interval = 24  # sample every 24 frames
        count = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            _, buffer = cv2.imencode(".jpg", frame)
            frame_b64 = base64.b64encode(buffer).decode("utf-8")
            base64_frames.append(frame_b64)
            # Save sampled frame for preview grid
            if count % frame_interval == 0:
                sampled_images.append(frame.copy())
            count += 1
        cap.release()
        print(f"{len(base64_frames)} frames extracted.")
        return base64_frames, sampled_images

    def analyze_frames(self, base64_frames):
        """Analyze frames using OpenAI."""
        action_prompt = """
### 基本分析
1. **动作识别**
    - "请识别图片中游泳者的泳姿类型（如自由泳、蛙泳等），并描述其基本动作。"
    - "游泳者的身体位置和姿态是什么？请描述其在水中的相对位置。"

2. **动作分解**
- "请详细描述图片中游泳者的头、手臂和腿部位置，并比较与标准泳姿的区别，指出可能存在的技术问题？"
- "这张图片中，游泳者的头部、手臂和腿部姿势是否正确？请分析并解释可能的技术误区。"

### 进阶分析
3. **技术建议**
- "基于此帧的分析，请提供改进游泳者技术的具体建议，特别是关于手臂划水和腿部蹬水的部分。"
- "请识别出游泳者可改进的技术细节，并建议合适的训练手段来加强这些方面。"

4. **动态分析**
- "设想游泳者从这帧开始的动作过渡，请提供可能的后续建议来优化整个游泳动作流畅性。"
- "在这张图片中，如何通过微调姿态改善整体动力？例如，是否需要调整入水角度或者蹬腿节奏？"

5. **互动建议**
- "如果你有任何想法或建议来即刻改善这个姿势，从而提升游泳技术动作，请描述详情。"
- "如果现有的体位调整能改善这帧中的姿态，请描述具体如何执行这类调整。"
"""
        prompt_messages = [
            {
                "role": "user",
                "content": [
                    action_prompt,
                    *list(map(lambda x: {"image": x}, base64_frames[0::24]))
                ],
            },
        ]

        params = {
            "model": "gpt-4o-2024-11-20",
            "messages": prompt_messages,
            "max_tokens": 4096,
        }

        result = self.client.chat.completions.create(**params)
        return result.choices[0].message.content

    def save_sampled_frames(self, sampled_images):
        """Save sampled frames as image files and return file paths."""
        sampled_paths = []
        for i, frame in enumerate(sampled_images):
            path = os.path.join(self.output_dir, f"frame_{i}.jpg")
            cv2.imwrite(path, frame)
            sampled_paths.append(path)
        print(f"Saved {len(sampled_paths)} frames to '{self.output_dir}'")
        return sampled_paths

    def process_video(self, video_file):
        """Process the video and return the analysis result and grid images."""
        base64_frames, sampled_images = self.extract_frames(video_file)
        frame_grid_paths = self.save_sampled_frames(sampled_images)
        # NOTE: 这里只返回分析结果和图片路径，便于后续gradio多输出
        result = self.analyze_frames(base64_frames)
        return result, frame_grid_paths

def upload_video(video_file):
    """Gradio interface function for video upload and analysis."""
    analysis_tool = SwimAnalysisTool()
    markdown_result, image_grid_paths = analysis_tool.process_video(video_file)
    return markdown_result, image_grid_paths

# -------- Gradio 改进部分 --------
# 1. 输出 markdown文本
markdown_output = gr.Markdown(label="Analysis", elem_id="result_markdown")
# 2. 图像采样网格
sampled_grid = gr.Gallery(label="Sample frame preview", columns=4, height=240)
# 3. input框（gr.Video）固定宽高, 不会跟上传视频比例变化而撑大
fixed_video_input = gr.Video(label="Upload a swimming video", height=350, width=480)

iface = gr.Interface(
    fn=upload_video,
    inputs=fixed_video_input,
    outputs=[markdown_output, sampled_grid],
    title="Swim Technique Analysis",
    description="Upload a swimming video and get an analysis of the swimmer's technique.",
    theme=gr.themes.Monochrome(),
    css="footer {visibility: hidden}"
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", debug=True, server_port=7860)