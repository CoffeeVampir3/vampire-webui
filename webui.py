import gradio as gr
import sys
import random
import diffusion_pipeline as dp
import ui_config

def randomize_seed():
    return random.randint(0, (sys.maxsize/64))

def auto_seed():
    return -1

def model_changed(new_model):
    dp.load_pipeline(new_model)
    print(f'Loaded {new_model}')

css = """
        #seedbox {width: 30%; justify-content:space-between;}
        #seedsliders {min-width: 67%; flex-grow: 1; background: transparent; border-style:none;}
        #gallerybox {background: transparent; border-style:none;}
        #noborder {border-style:none; border-color: transparent;}
        #model_dropdown {width:30%; background: transparent; justify-content:space-between;}
    """

ui_config.enumerate_models()

app = gr.Blocks(css=css)
with app:
    with gr.Row():
        with gr.Column():
            prompt_textbox = gr.Textbox(lines=3, placeholder="Promp here", label="Prompt")
            negative_prompt_textbox = gr.Textbox(lines=3, placeholder="Negatives here", label="Negative")
            
            with gr.Box():
                with gr.Row(elem_id="noborder"):
                    with gr.Box(elem_id="seedbox"):
                        seed = gr.Number(value=1,label="Seed",precision=0, elem_id="noborder")
                        random_btn = gr.Button(value="Rand")
                        random_btn.click(fn=randomize_seed, inputs=None, outputs=seed)
                        auto_seed_btn = gr.Button(value="Auto")
                        auto_seed_btn.click(fn=auto_seed, inputs=None, outputs=seed)
                    with gr.Box(elem_id="seedsliders"):
                        in_parallel_slider = gr.Slider(minimum=1, maximum=100,value=1,step=1, label="Images Per Batch", elem_id="noborder")
                        generation_runs_slider = gr.Slider(minimum=1, maximum=100,value=1,step=1, label="Number of Batches", elem_id="noborder")

            width_slider = gr.Slider(minimum=256, maximum=2048,value=1,step=32, label="Width")
            height_slider = gr.Slider(minimum=256, maximum=2048,value=1,step=32, label="Height")
            num_steps_slider = gr.Slider(minimum=1, maximum=250,value=1,step=1, label="Steps")
            cfg_slider = gr.Slider(minimum=1, maximum=50,value=1,step=0.5, label="CFG Scale")

        with gr.Box(elem_id="gallerybox"):
            output_img = gr.Gallery()
            with gr.Row():
                model_dropdown = gr.Dropdown(label="Model", choices=ui_config.enumerate_models())
                model_dropdown.change(fn=model_changed, inputs=model_dropdown, outputs=None)

                app_inputs = [model_dropdown, prompt_textbox, negative_prompt_textbox, seed, in_parallel_slider, generation_runs_slider, width_slider, height_slider, num_steps_slider, cfg_slider]
                launch_btn = gr.Button(value="Generate")
                launch_btn.click(fn=dp.run_pipeline, inputs=app_inputs, outputs=output_img)

        app.load(ui_config.load_ui_config, inputs=None, outputs=app_inputs)

app.launch()