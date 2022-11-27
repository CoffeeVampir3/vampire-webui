import gradio as gr
import ui.txt2img_ui
import ui.convert_to_diffuser_ui

css = """
        #seedbox {width: 30%; justify-content:space-between;}
        #seedsliders {min-width: 67%; flex-grow: 1; background: transparent; border-style:none;}
        #gallerybox {background: transparent; border-style:none;}
        #noborder {border-style:none; border-color: transparent;}
        #model_dropdown {width:30%; background: transparent; justify-content:space-between;}
    """

app = gr.Blocks(css=css)
with app:
    with gr.Tab(label="Text to Image"):
        ui.txt2img_ui.create(css)
    #with gr.Tab(label="Upgrade old models to diffusers"):
    #    ui.convert_to_diffuser_ui.create(css)

app.launch(enable_queue=True)