import glfw
from OpenGL.GL import *
from OpenGL.GL import shaders
import imgui
from imgui.integrations.glfw import GlfwRenderer
from PIL import Image
import numpy as np
import time
import imageio

def impl_glfw_init():
    window_name = 'A Simple Animation Program'

    if not glfw.init():
        print('[-] could not initialize OpenGL context')
        exit(1)

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)

    # Create a window and its OpenGL context
    window = glfw.create_window(
        int(width), int(height), window_name, None, None)
    glfw.make_context_current(window)

    if not window:
        glfw.terminate()
        print('[-] could not initialize window')
        exit(1)

    return window

def on_key(window, key, scancode, action, mods):
    global active_fg_element
    if key == glfw.KEY_TAB and action == glfw.RELEASE:
        global hide_panel
        hide_panel = not hide_panel
    elif key == glfw.KEY_SPACE and action == glfw.RELEASE:
        # Reset the trace buffer, so that another can be recorded
        del traces[active_fg_element][:]  
        del trace_timestamps[active_fg_element][:]
    elif key == glfw.KEY_S and action == glfw.RELEASE:
        # Switch the active fg element
        active_fg_element = (active_fg_element + 1) % len(fg_elements)

def on_cursor_move(window, xpos, ypos):
    if trace_active:
        traces[active_fg_element].append((int(xpos), int(ypos)))
        sec_elapsed = time.time() - trace_start_time
        trace_timestamps[active_fg_element].append(sec_elapsed)

def on_mouse_button(window, button, action, mods):
    global mouse_pressed
    global trace_active
    global trace_start_time
    if button == glfw.MOUSE_BUTTON_LEFT:
        if action == glfw.PRESS:
            mouse_pressed = True
            if len(traces[active_fg_element]) == 0:
                trace_active = True
                trace_start_time = time.time()
        elif action == glfw.RELEASE:
            mouse_pressed = False
            trace_active = False

def load_image(path):
    im = Image.open(path).transpose(Image.FLIP_TOP_BOTTOM)
    im_data = np.array(im.getdata(), np.uint8)
    im_data = im_data.astype(np.float) / 255.0
    im_width, im_height = im.size
    im_channels = im_data.shape[-1]
    return im_data.reshape((im_height, im_width, im_channels))

def get_tex_format(n_channels):
    if n_channels == 1:
        return GL_RED
    elif n_channels == 2:
        return GL_RG
    elif n_channels == 3:
        return GL_RGB
    elif n_channels == 4:
        return GL_RGBA
    else:
        return None

def load_texture(im_data):
    im_height, im_width, im_channels = im_data.shape
    tex_format = get_tex_format(im_channels)

    texture = glGenTextures(1)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexImage2D(GL_TEXTURE_2D, 0, tex_format, im_width, im_height, 0, tex_format, GL_FLOAT, im_data)
    glGenerateMipmap(GL_TEXTURE_2D)
    return texture

def update_texture(im_data):
    im_height, im_width, im_channels = im_data.shape
    tex_format = get_tex_format(im_channels)
    
    out_data = np.copy(im_data)
    if not trace_active:
        for i in range(len(fg_elements)):
            trace = traces[i]
            if len(trace) > 0:
                trace_idxs[i] = (trace_idxs[i] + 1) % len(trace)
                x, y = trace[trace_idxs[i]]

                fg_im_data = fg_elements[i]
                fg_h, fg_w, fg_c = fg_im_data.shape
                y0 = max(height - y - fg_h // 2, 0)
                y1 = min(y0 + fg_h, im_height)
                x0 = max(x - fg_w // 2, 0)
                x1 = min(x0 + fg_w, im_width)
                if y1 - y0 > 0 and x1 - x0 > 0:
                    fg_alpha = fg_im_data[:y1-y0, :x1-x0, 3:]
                    out_data[y0:y1, x0:x1, :] *= 1.0 - fg_alpha
                    out_data[y0:y1, x0:x1, :] += fg_alpha * fg_im_data[:y1-y0, :x1-x0, :3]

    glBindTexture(GL_TEXTURE_2D, texture)
    glTexImage2D(GL_TEXTURE_2D, 0, tex_format, im_width, im_height, 0, tex_format, GL_FLOAT, out_data)

def prepare_shader():
    global bg_im_data

    vertex_shader = shaders.compileShader("""#version 330
    layout(location = 0) in vec2 position;
    layout(location = 1) in vec2 texCoord;
    out vec2 vTexCoord;
    void main() {
        vTexCoord = texCoord;
        gl_Position = vec4(position, 0, 1);
    }""", GL_VERTEX_SHADER)

    fragment_shader = shaders.compileShader("""#version 330
    uniform sampler2D tex;
    in vec2 vTexCoord;
    out vec4 fragColor;
    void main() {
        fragColor = texture(tex, vTexCoord);
    }""", GL_FRAGMENT_SHADER)

    vertex_data = np.array([
        #  X,    Y,   U,   V,
        -1.0, -1.0, 0.0, 0.0,
         1.0, -1.0, 1.0, 0.0,
        -1.0,  1.0, 0.0, 1.0,
        -1.0,  1.0, 0.0, 1.0,
         1.0, -1.0, 1.0, 0.0,
         1.0,  1.0, 1.0, 1.0,
    ], dtype=np.float32)

    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)

    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_STATIC_DRAW)

    shader = shaders.compileProgram(vertex_shader, fragment_shader)
    positionAttrib = glGetAttribLocation(shader, 'position')
    coordsAttrib   = glGetAttribLocation(shader, 'texCoord')

    glEnableVertexAttribArray(0)
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(positionAttrib, 2, GL_FLOAT, GL_FALSE, 16, None)
    glVertexAttribPointer(coordsAttrib,   2, GL_FLOAT, GL_TRUE,  16, ctypes.c_void_p(8))

    bg_im_data = load_image('images/bokeh.jpg')
    texture = load_texture(bg_im_data)
    texture_locn = glGetUniformLocation(shader, 'tex')

    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)

    return shader, VAO, texture, texture_locn

def add_fg_element(im_data):
    fg_elements.append(im_data)
    traces.append([])
    trace_timestamps.append([])
    trace_idxs.append(0)
    fg_element_labels.append('element%d' % len(fg_elements))
    fg_colors.append(np.random.random(3))

    global active_fg_element
    active_fg_element = len(fg_elements) - 1

def gaussian_spray(width, height, sigma_x, sigma_y):
    # 2D Gaussian formula:
    # https://en.wikipedia.org/wiki/Gaussian_function#Two-dimensional_Gaussian_function
    x = np.arange(-width // 2, -width // 2 + width)
    y = np.arange(-height // 2, -height // 2 + height)
    xx, yy = np.meshgrid(x, y)
    term0 = np.square(xx) / np.square(sigma_x)
    term1 = np.square(yy) / np.square(sigma_y)
    probs = np.exp(-0.5 * (term0 + term1))
    rand = np.random.random(probs.shape)
    return rand < probs

def export_gif():
    global exporting
    if not exporting:
        exporting = True
        out_path = 'output.gif'
        im_data = np.copy(bg_im_data)
        with imageio.get_writer(out_path, mode='I', fps=50) as writer:
            n_frames = max([len(t) for t in traces])
            for k in range(n_frames):
                print('[+] exporting frame %d of %d' % (k + 1, n_frames))
                im_height, im_width, im_channels = im_data.shape
                out_data = np.copy(im_data)
                for i in range(len(fg_elements)):
                    trace = traces[i]
                    if len(trace) > 0:
                        x, y = trace[k % len(trace)]
                        fg_im_data = fg_elements[i]
                        fg_h, fg_w, fg_c = fg_im_data.shape
                        y0 = max(height - y - fg_h // 2, 0)
                        y1 = min(y0 + fg_h, im_height)
                        x0 = max(x - fg_w // 2, 0)
                        x1 = min(x0 + fg_w, im_width)
                        if y1 - y0 > 0 and x1 - x0 > 0:
                            fg_alpha = fg_im_data[:y1-y0, :x1-x0, 3:]
                            out_data[y0:y1, x0:x1] *= 1.0 - fg_alpha
                            out_data[y0:y1, x0:x1] += fg_alpha * fg_im_data[:y1-y0, :x1-x0, :3]
                            if spray_trail:
                                split = 4
                                sy0 = y0  + (y1 - y0) // (split * 2)
                                sy1 = sy0 + (y1 - y0) // (split)
                                sx0 = x0  + (x1 - x0) // (split * 2)
                                sx1 = sx0 + (x1 - x0) // (split)
                                spray_h, spray_w = out_data[sy0:sy1, sx0:sx1].shape[:2]
                                spray = gaussian_spray(spray_w, spray_h, float(spray_w) / 2, float(spray_h) / 2)
                                im_data[sy0:sy1, sx0:sx1, :3][spray] = fg_colors[i]  # splatter trail
                out_data = (out_data * 255).astype(np.uint8)
                writer.append_data(out_data[::-1])  # flip y-axis
        print('[+] exported to %s' % out_path)
        exporting = False

hide_panel = False
mouse_pressed = False
width, height = 1280, 720
traces = []  # list of trace lists
trace_timestamps = []  # list of timestamp lists
trace_idxs = []  # list of current indices, one for each trace
trace_active = False
trace_start_time = 0
fg_element_labels = []
fg_colors = []
active_fg_element = None
bg_im_data = None
fg_elements = []
exporting = False
timestep = 0
spray_trail = True

if __name__ == '__main__':
    imgui.create_context()
    window = impl_glfw_init()
    impl = GlfwRenderer(window)

    glfw.set_key_callback(window, on_key)
    glfw.set_cursor_pos_callback(window, on_cursor_move)
    glfw.set_mouse_button_callback(window, on_mouse_button)
    shader, VAO, texture, texture_locn = prepare_shader()

    imgui.set_window_focus()

    add_fg_element(load_image('images/heli.png'))
    add_fg_element(load_image('images/heli.png'))
    active_fg_element = 0

    while not glfw.window_should_close(window):
        glfw.poll_events()
        impl.process_inputs()

        imgui.new_frame()
        curr_width, curr_height = glfw.get_window_size(window)
        if curr_width != width or curr_height != height:
            width, height = curr_width, curr_height
        
        if not hide_panel:
            # Timeline
            timeline_height = 80
            imgui.set_next_window_position(0, height - timeline_height)
            imgui.set_next_window_size(width, timeline_height)
            imgui.set_next_window_bg_alpha(0.8)
            imgui.begin('timeline', False, imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE)
            changed, timestep = imgui.slider_int(
                'timestep', timestep, min_value=0, max_value=100, format='%d')
            if imgui.button('Play Back Animation'):
                print('[TODO: play back the animation]')
            imgui.end()

            # Control panel
            panel_width = max(width // 4, 300)
            imgui.set_next_window_position(width - panel_width, 0)
            imgui.set_next_window_size(panel_width, height - timeline_height)
            imgui.set_next_window_bg_alpha(0.8)
            imgui.begin('controlpanel', False, imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE)
            imgui.text('Control Panel')
            imgui.dummy(0, 10)
            imgui.text('active fg element')
            clicked, active_fg_element = imgui.listbox(
                '', active_fg_element, fg_element_labels)
            if imgui.button('Export GIF'):
                export_gif()
            imgui.end()

        glClearColor(1.0, 1.0, 1.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        glUseProgram(shader)
        try:
            glActiveTexture(GL_TEXTURE0)
            if not trace_active:
                update_texture(bg_im_data)
            glBindTexture(GL_TEXTURE_2D, texture)
            glUniform1i(texture_locn, 0)
            glBindVertexArray(VAO)
            glDrawArrays(GL_TRIANGLES, 0, 6)
        finally:
            glBindVertexArray(0)
            glUseProgram(0)

        imgui.render()
        impl.render(imgui.get_draw_data())
        glfw.swap_buffers(window)

    impl.shutdown()
    glfw.terminate()
