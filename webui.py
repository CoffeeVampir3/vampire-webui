import gradio as gr
import diffusion_pipeline as dp

prompt_textbox = gr.Textbox(lines=3, placeholder="")
negative_prompt_textbox = gr.Textbox(lines=3, placeholder="")
seed = gr.Number(value=12345,label="Seed",precision=0)
num_inputs_slider = gr.Slider(minimum=1, maximum=100,value=1,step=1)
width_slider = gr.Slider(minimum=256, maximum=2048,value=512,step=64)
height_slider = gr.Slider(minimum=256, maximum=2048,value=512,step=64)
num_steps_slider = gr.Slider(minimum=1, maximum=250,value=1,step=1)
cfg_slider = gr.Slider(minimum=1, maximum=50,value=7,step=0.5)

output_img = gr.Gallery()

pipeline = gr.State(value = dp.load_pipeline)
app_inputs = [pipeline, prompt_textbox, negative_prompt_textbox, seed, num_inputs_slider, width_slider, height_slider, num_steps_slider, cfg_slider]
app = gr.Interface(fn=dp.testpipeline, inputs=app_inputs, outputs=[output_img, pipeline], allow_flagging="never")
app.launch()
