import gradio as gr
import diffusion_pipeline as dp


num_images = 1
prompt = ["Woman riding a horse, moonlight, oil painting on canvas"] * num_images

prompt_textbox = gr.inputs.Textbox(lines=3, placeholder="")
num_inputs_slider = gr.inputs.Slider(minimum=1, maximum=100,default=1,step=1)
width_slider = gr.inputs.Slider(minimum=256, maximum=2048,default=512,step=64)
height_slider = gr.inputs.Slider(minimum=256, maximum=2048,default=512,step=64)
num_steps_slider = gr.inputs.Slider(minimum=1, maximum=250,default=1,step=1)
cfg_slider = gr.inputs.Slider(minimum=1, maximum=30,default=7,step=0.5)

output_img = gr.Gallery(type="pil")

app_inputs = [prompt_textbox, num_inputs_slider, width_slider, height_slider, num_steps_slider, cfg_slider]
app = gr.Interface(fn=dp.testpipeline, inputs=app_inputs, outputs=output_img)
app.launch()