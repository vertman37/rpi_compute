import pyglet

import platform
if platform.system() == "Linux":
    with open("/proc/cpuinfo") as f:
        cpuinfo = f.read()
    if "Raspberry" in cpuinfo:  # Broadcom is Raspberry Pi
        print("Raspberry Pi detected")
        # Disable shadow window
        pyglet.options.shadow_window = False

# Query the best config for the screen
display = pyglet.display.get_display()
screen = display.get_default_screen()
config = screen.get_best_config()

# Specify that we want to use OpenGL ES 3.1
# This is very specific to the Raspberry Pi 4 and 5. Use 3.2 if you can.
config.opengl_api = "gles"
config.major_version = 3
config.minor_version = 1

# Create the window
window = pyglet.window.Window(config=config)

import inspect
from OpenGL.GL.shaders import compileShader, compileProgram
#https://pyglet.readthedocs.io/en/development/programming_guide/opengles.html

from pyglet.gl import * #somehow, it should be after window.

# from OpenGL.GL import * #try to avoid it to rpi, since not glDispatchCompute
# from pyglet.gl import glDispatchCompute,glMemoryBarrier,glVertexAttribPointer#....
#glPointSize not working in rpi
print('compute',bool(glDispatchCompute))

value = GLint()
glGetIntegerv(GL_MAX_COMPUTE_SHARED_MEMORY_SIZE, value)
print(value, "is shared memory size")
# exit()

#not this way.. pyglet has diff.implement.
# buffer = glGenBuffers(1)
# glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer)
# glBufferData(GL_SHADER_STORAGE_BUFFER, data.nbytes, data, GL_DYNAMIC_DRAW)
# glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buffer)

#we cant DSA buffer..
# buffer = ctypes.c_uint()
# glCreateBuffers(1, ctypes.byref(buffer))
# glNamedBufferStorage(buffer, size, data, flag)


#seems not in rpi
# ptr = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY)

#glGetBufferSubData not supported while glGetNamedBufferSubData /bindless buffer,too.
# rdata = np.zeros(16, dtype=np.float32)  # 0~15
# ptr = rdata.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
# bytes_arr = glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, rdata.nbytes, ptr) #<class 'numpy.ndarray'>



import numpy as np
import ctypes
import math

class SSBO:
    def __init__(self, data):
        # ssbo = GLuint() #maybe unstable, use ctypes instead.
        ssbo = ctypes.c_uint()
        glGenBuffers(1, ctypes.byref(ssbo))
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo)
        glBufferData(GL_SHADER_STORAGE_BUFFER, data.nbytes, data.ctypes.data, GL_STATIC_DRAW)#for pyglet
        # glBufferData(GL_SHADER_STORAGE_BUFFER, data.nbytes, data, GL_STATIC_DRAW)
        self.ID = ssbo
        self.size = data.size
        self._rdata = None
    
    def bind(self,channel):
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, channel, self.ID)  # binding=0
    
    def read(self):
        if self._rdata is None:
            self._rdata = np.zeros(self.size, dtype=np.float32)
        rdata = self._rdata
        
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.ID)#required before map!
        ptr = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, rdata.nbytes, GL_MAP_READ_BIT)
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER)
        
        # float_array = (ctypes.c_float*rdata.size).from_address(ptr)
        # read_data = np.frombuffer(float_array, dtype=np.float32)
        read_data = np.ctypeslib.as_array(ctypes.cast(ptr, ctypes.POINTER(ctypes.c_float)), shape=(rdata.size,))
        return read_data

    def compute_mul(self, val):        
        glUseProgram(program_mul)
        self.bind(0)
        self._dispatch()
    def compute_muladd(self, val,mul):
        # Fused Multiply-Add (FMA) ,  fma(a, b, c); // a*b + c / pos = fma(spd, dt, pos); // pos + spd*dt 
        glUseProgram(program_muladd)
        self.bind(0)
        val.bind(1)
        self._dispatch()    
    
    def _dispatch(self):    
        N = self.size
        dispatch_size = math.ceil(N/BLOCKSIZE)
        glDispatchCompute(dispatch_size,1,1)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)  # sync for SSBO


#RPI4, MAX_COMPUTE_WORK_GROUP_SIZE = 256
BLOCKSIZE = 256
compute_src = """
#version 310 es
precision highp float;

layout(local_size_x = 16) in;

layout(std430, binding = 0) buffer Data {
    float values[];
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    values[idx] *= 2.0;  //mere double
}
"""
program_mul = compileProgram(compileShader(inspect.cleandoc(compute_src), GL_COMPUTE_SHADER))

compute_src = """
#version 310 es
precision highp float;

layout(local_size_x = 256) in;

layout(std430, binding = 0) buffer Data {
    float pos[];
};

layout(std430, binding = 1) buffer Data2 {
    float spd[];
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    pos[idx] += spd[idx] * 0.1;
}
"""
program_muladd = compileProgram(compileShader(inspect.cleandoc(compute_src), GL_COMPUTE_SHADER))

# pos = np.arange(100, dtype=np.float32)  # 0~15
# pos_ssbo = SSBO(pos)
# spd = np.ones(100, dtype=np.float32)
# spd_ssbo = SSBO(spd)

# pos_ssbo.compute_muladd(spd_ssbo, 0.01)
# print(pos_ssbo.read())


N = 100000
pos = np.random.rand(N*3).astype(np.float32)-0.5
pos = np.zeros(N*3, dtype=np.float32)
pos[2::3] = 0

spd = np.random.rand(N*3).astype(np.float32)-0.5
spd /= 10
# spd_x = spd[::3]
# spd_y = spd[1::3]
# spd_z = spd[2::3]
spd[2::3] = 0

pos_ssbo = SSBO(pos)
spd_ssbo = SSBO(spd)

pos_ssbo.compute_muladd(spd_ssbo, 0.01)
print(pos_ssbo.read())




# ssbo = ctypes.c_uint()# ssbo = GLuint()
# glGenBuffers(1, ctypes.byref(ssbo))
# glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo)
# glBufferData(GL_SHADER_STORAGE_BUFFER, data.nbytes, data.ctypes.data, GL_DYNAMIC_COPY)
# glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo)  # binding=0

# read_data = np.frombuffer(bytes_arr, dtype=np.float32)
# print(read_data)

# ptr = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY)
# print(ptr)
# glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, data.nbytes, GL_MAP_READ_BIT)#나쁘지않음

# result = np.frombuffer((ctypes.c_float * 16).from_address(ctypes.addressof(ptr.contents)), dtype=np.float32)
# glUnmapBuffer(GL_SHADER_STORAGE_BUFFER)

x = """
layout(std430, binding = 0) buffer DataBuffer {
    float values[];
};
layout(std140) uniform DataBuffer {
    float values[16];
};
"""

#no SSBO for vert!!!!
vert_ssbo = """
#version 310 es
precision highp float;

layout(std430, binding = 0) buffer DataBuffer {
    float values[];
};

void main(void){
    //gl_Position = vec4(0.5,0,0,1);
    float x = values[gl_VertexID];
    gl_Position = vec4(x,0,0,1);
}
"""

vert = """
#version 310 es
precision highp float;

layout(location = 0) in vec3 position;
//layout(location = 0) in float x;

void main() {
    gl_Position = vec4(position, 1.0);
    //gl_Position = vec4(x*0.01,0,0, 1.0);
}
"""
frag = """
#version 310 es
precision highp float;

out vec4 color;

void main(void){
    color = vec4(1,0,1, 1.0);
}
"""
# from OpenGL.GL.shaders import compileShader, compileProgram
sha = compileProgram(compileShader(inspect.cleandoc(vert), GL_VERTEX_SHADER), compileShader(inspect.cleandoc(frag), GL_FRAGMENT_SHADER))
glUseProgram(sha)

# block_index = glGetUniformBlockIndex(sha, "DataBuffer")ERROR

# vao = ctypes.c_uint()
# glGenVertexArrays(1, ctypes.byref(vao))
# glBindVertexArray(vao)

vao = GLuint()
glGenVertexArrays(1, vao)
glBindVertexArray(vao)

glBindBuffer(GL_ARRAY_BUFFER, pos_ssbo.ID)
glEnableVertexAttribArray(0)
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, pos_ssbo.ID)#has already done

# glEnable(GL_PROGRAM_POINT_SIZE)
try:
    glPointSize(5)
except:
    pass
glClearColor(0.2,0.2,0.2,1)

import time
@window.event
def on_draw():
    # window.clear()
    glClear(GL_COLOR_BUFFER_BIT)

    # glUseProgram(program)
    # glDispatchCompute(1,1,1)
    # glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

    pos_ssbo.compute_muladd(spd_ssbo, 0.01)
    glUseProgram(sha)
    glDrawArrays(GL_POINTS, 0, N)
    time.sleep(0.01)
    # glDrawArrays(GL_TRIANGLES, 0, 6)
    
pyglet.app.run()

# while not glfw.window_should_close(window):    
#     glClear(GL_COLOR_BUFFER_BIT)
    
#     glDrawArrays(GL_POINTS, 0, 3)

#     glfw.poll_events()
#     glfw.swap_buffers(window)

