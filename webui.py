import gradio as gr
import diffusion_pipeline as dp
import ui_config

config = ui_config.load_ui_config()

prompt_textbox = gr.Textbox(lines=3, value=config.prompt)
negative_prompt_textbox = gr.Textbox(lines=3, value=config.neg_prompt)
seed = gr.Number(value=config.seed,label="Seed",precision=0)
in_parallel_slider = gr.Slider(minimum=1, maximum=100,value=config.generate_x_in_parallel,step=1)
generation_runs_slider = gr.Slider(minimum=1, maximum=100,value=config.batches,step=1)
width_slider = gr.Slider(minimum=256, maximum=2048,value=config.width,step=32)
height_slider = gr.Slider(minimum=256, maximum=2048,value=config.height,step=32)
num_steps_slider = gr.Slider(minimum=1, maximum=250,value=config.num_steps,step=1)
cfg_slider = gr.Slider(minimum=1, maximum=50,value=config.cfg,step=0.5)

output_img = gr.Gallery()
app_inputs = [prompt_textbox, negative_prompt_textbox, seed, in_parallel_slider, generation_runs_slider, width_slider, height_slider, num_steps_slider, cfg_slider]

txt_to_img_interface = gr.Interface(fn=dp.run_pipeline, inputs=app_inputs, outputs=output_img, allow_flagging="never")

app = gr.Blocks()
with app:
    gr.Interface(fn=dp.run_pipeline, inputs=app_inputs, outputs=output_img, allow_flagging="never")

#dp.load_pipeline()
#app = gr.TabbedInterface([txt_to_img_interface, test_button_interface])
#app = gr.Interface(fn=dp.run_pipeline, inputs=app_inputs, outputs=output_img, allow_flagging="never")
app.launch()
