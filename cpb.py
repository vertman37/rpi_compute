import time
import numpy as np
import ctypes
import math
from pyrr import matrix44

import inspect
from OpenGL.GL.shaders import compileShader, compileProgram
#https://pyglet.readthedocs.io/en/development/programming_guide/opengles.html
from OpenGL.GL import *





#=================pygletforRPI4=================
import pyglet
pyglet.options.shadow_window = False
# from pyglet.gl import * #somehow, it should be after window.

import platform
if platform.system() == "Linux":
    with open("/proc/cpuinfo") as f:
        cpuinfo = f.read()
    if "Raspberry" in cpuinfo:  # Broadcom is Raspberry Pi
        print("Raspberry Pi detected")
        # Disable shadow window

        #override False functions.
        # from pyglet.gl import glDispatchCompute,glMemoryBarrier,glProgramUniformMatrix4fv
        import pyglet.gl as _gl
        glDispatchCompute = _gl.glDispatchCompute
        glMemoryBarrier = _gl.glMemoryBarrier
        glVertexAttribPointer = _gl.glVertexAttribPointer
        
        # glGetUniformLocation = _gl.glGetUniformLocation
        # glProgramUniformMatrix4fv = _gl.glProgramUniformMatrix4fv
        
        #why? forgot the reason.
        # glBindBufferBase = _gl.glBindBufferBase
        # def glBindBufferBase(target, index, buffer):
        #     index = ctypes.c_uint(index).value
        #     buffer = ctypes.c_uint(buffer).value
        #     _gl.glBindBufferBase(target,index,buffer)

        def glPointSize(value):
            return

        #DSA but no option.
        #THIS WAS NOT SSBO BINDING LOCATION!!! just found index.
        def glGetProgramResourceIndex(program,target,name):
            name = bytes(name,encoding='utf-8')
            return _gl.glGetProgramResourceIndex(program,target,name)
        
        #not supported in ES.
        # glShaderStorageBlockBinding
        
        #this is DSA. use old way instead.
        # def glProgramUniformMatrix4fv(program,location,count,transpose,value):
        #     value = np.asarray(value)
        #     value = value.astype(np.float32).flatten()
        #     value = value.ctypes.data_as(ctypes.POINTER(ctypes.c_float))#rpi requires
        #     _gl.glProgramUniformMatrix4fv(program,location,count,transpose,value)


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
#=================pygletforRPI4=================





#check the GL functions
#glPointSize not working in rpi
print('glDispatchCompute',bool(glDispatchCompute))
print('glProgramUniformMatrix4fv',bool(glProgramUniformMatrix4fv))
print("GL_MAX_COMPUTE_SHARED_MEMORY_SIZE", glGetIntegerv(GL_MAX_COMPUTE_SHARED_MEMORY_SIZE))
print("GL_MAX_COMPUTE_WORK_GROUP_SIZE", glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE,0))




# BUFFER
# buffer = glGenBuffers(1)
# glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer)
# glBufferData(GL_SHADER_STORAGE_BUFFER, data.nbytes, data, GL_DYNAMIC_DRAW)
# glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buffer)

# DSA BUFFER. rpi not supported.
# buffer = ctypes.c_uint()
# glCreateBuffers(1, ctypes.byref(buffer))
# glNamedBufferStorage(buffer, size, data, flag)

# MAPPING. seems not in rpi
# ptr = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY)
# ptr = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, data.nbytes, GL_MAP_READ_BIT)

#glGetBufferSubData not supported while glGetNamedBufferSubData -for DSA buffer,too.
# rdata = np.zeros(16, dtype=np.float32)  # 0~15
# ptr = rdata.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
# bytes_arr = glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, rdata.nbytes, ptr) #<class 'numpy.ndarray'>


class SSBO:
    def __init__(self, data):
        ssbo = glGenBuffers(1)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo)
        glBufferData(GL_SHADER_STORAGE_BUFFER, data.nbytes, data, GL_STATIC_DRAW)
        self.ID = ssbo
        self.size = data.size
        self._rdata = None

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


#RPI4, GL_MAX_COMPUTE_WORK_GROUP_SIZE = 256
BLOCKSIZE = 256

compute_src_posspd ="""
#version 310 es
precision highp float;

uniform float dt;

layout(local_size_x = 256) in;

layout(std430, binding=0) buffer Pos {
    float pos[];
};

layout(std430, binding=1) buffer Spd {
    float spd[];
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    pos[idx] += spd[idx] * dt;
}
"""


class Shader:
    def __init__(self):
        self.loc_ssbo = {}        
        self.loc_uniform = {}
        #set initial default value.. which shall overwriten after,huh.
        # for key,value in kwargs.items():
        #     loc = glGetUniformLocation(program, key)
        #     glProgramUniform1f(program, loc, value)#for initial value,DSA.
        #     self.loc_uniform[key] = loc
    
    def use(self):
        glUseProgram(self.program)
    
    def get_ubo_index(self, name):
        block_index = glGetUniformBlockIndex(program_muladd, "Data")#UBO

    def get_loc_ssbo(self, name):
        try:
            loc = self.loc_ssbo[name]
        except KeyError:
            #index is just found counter. not the location / binding point!
            idx = glGetProgramResourceIndex(self.program, GL_SHADER_STORAGE_BLOCK, name)

            #we force to use binding=0,1,2.. so that found counter will matched.
            loc = idx
            #it gets binding=x to x. without binding, it returns 0.
            # print(glGetProgramResourceiv(sha_posspd.program, GL_SHADER_STORAGE_BLOCK, 0, 1,[GL_BUFFER_BINDING], 1))
            
            #finally found: with out binding, explicit setting is required.
            #but not supported in GL ES3.1.
            # loc = len(self.loc_ssbo)
            # glShaderStorageBlockBinding(self.program, idx, loc)
            
            self.loc_ssbo[name] = loc
        return loc
    
    def get_loc_uniform(self, name):
        try:
            loc = self.loc_uniform[name]
        except KeyError:
            loc = glGetUniformLocation(self.program, name)
            self.loc_uniform[name] = loc
        return loc

    def bind_ssbo(self, name, ssbo):
        idx = self.get_loc_ssbo(name)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, idx, ssbo.ID)

    def set_uniform1(self, name,value):
        loc = self.get_loc_uniform(name)
        self.use()#for future, it will be removed. only for RPI,legacy.
        glUniform1f(loc, value)  #set variable
        # glProgramUniform1f / DSA
    
    def set_uniform4x4(self, name,value):
        loc = self.get_loc_uniform(name)
        #glProgramUniformMatrix4fv(self.program, loc, 1, GL_FALSE, value)#DSA style!
        self.use()
        glUniformMatrix4fv(loc, 1, GL_FALSE, value)



class ComputeShader(Shader):
    def __init__(self, compute_src):
        super().__init__()
        self.program = compileProgram(compileShader(inspect.cleandoc(compute_src), GL_COMPUTE_SHADER))        

    def dispatch(self,N):
        # "make sure to use()!"
        self.use()#made it sure.
        dispatch_size = math.ceil(N/BLOCKSIZE)
        glDispatchCompute(dispatch_size,1,1)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)  # sync for SSBO

    def execute(self, N, **kwargs):
        #without N? however..
        for key,ssbo in kwargs.items():
            self.bind_ssbo(key,ssbo)
        self.dispatch(N)

class DrawShader(Shader):
    def __init__(self, vs,fs, gs=None, size=3, stride=0):
        super().__init__()
        self.program = compileProgram(compileShader(inspect.cleandoc(vs), GL_VERTEX_SHADER), compileShader(inspect.cleandoc(fs), GL_FRAGMENT_SHADER))

    # def get_loc_in(self,name):
        # loc = glGetAttribLocation(self.program, name)
        #DSA. for future.
        # idx = glGetProgramResourceIndex(program, GL_PROGRAM_INPUT, "position");
        # glGetProgramResourceiv(program, GL_PROGRAM_INPUT, idx, 1, &prop, 1, NULL, &location);
        
    def draw(self,vao, **kwargs):
        "for future.. since without DSA, SSBO in vs, we use old way."
        self.use()        
        # glBindVertexArray(vao)
        # glBindBuffer(GL_ARRAY_BUFFER, ssbo.ID)
        #GL_ELEMENT_ARRAY_BUFFER
        # for key,ssbo in kwargs.items():
        vao.draw()



class VAO:
    "is for draw object. requires shader used first."
    def __init__(self, ssbo, size=3,stride=0):
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        glBindBuffer(GL_ARRAY_BUFFER, ssbo.ID)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, size, GL_FLOAT, GL_FALSE, stride, None)#index size type normalize stride offsetptr
        # ctypes.c_void_p(3 * vertices.itemsize)#offset is local.
        #stride = 5 * vertices.itemsize
        
        self.count = ssbo.size//size
        # if stride == 0:
            # self.count = ssbo.size//size
        # else:
        #     self.count = ssbo.size//stride
        # print(self.count,'count',size,ssbo.size)
    def draw(self):
        glBindVertexArray(self.vao)
        glDrawArrays(GL_POINTS, 0, self.count)
        # glDrawArrays(GL_TRIANGLES, 0, 6)



pos = np.arange(10, dtype=np.float32)  # 0~15
spd = np.ones(10, dtype=np.float32)
pos_ssbo = SSBO(pos)
spd_ssbo = SSBO(spd)

sha_posspd = ComputeShader(compute_src_posspd)
sha_posspd.set_uniform1("dt",0.1)

# sha_posspd.bind_ssbo("Pos",pos_ssbo)
# sha_posspd.bind_ssbo("Spd",spd_ssbo)
# sha_posspd.dispatch(pos_ssbo.size)
# or in short,
sha_posspd.execute(pos_ssbo.size, Pos=pos_ssbo,Spd=spd_ssbo)

print(pos_ssbo.read())
# exit()


#============part 2. 3d particles.
N = 10000
pos = np.zeros(N*3, dtype=np.float32)
spd = np.random.rand(N*3).astype(np.float32)-0.5
pos_ssbo = SSBO(pos)
spd_ssbo = SSBO(spd)

sha_posspd.execute(pos_ssbo.size, Pos=pos_ssbo,Spd=spd_ssbo)


UBO_or_SSBO = """
layout(location = 0) in float x;

layout(std430, binding = 0) buffer DataBuffer {
    float values[];
};

layout(std140) uniform DataBuffer {
    float values[16];
};
"""
#NOTE: RPI4 not allows SSBO in vertex shader.

vert = """
#version 310 es
precision highp float;

uniform mat4 ProjectionView;

layout(location = 0) in vec3 position;

void main() {
    gl_Position = ProjectionView * vec4(position, 1.0);
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

projection = matrix44.create_perspective_projection(45,16/9,0.01,100)#fov ratio near far
view = matrix44.create_look_at([3,3,3], [0,0,0], [0,1,0]) # view target up
projection_view = (projection.T@view.T).T #to pretty print(and gl-preffered), pyrr is col.major.

sha = DrawShader(vs=vert,fs=frag)
sha.set_uniform4x4("ProjectionView", projection_view)
# sha.set_in("position",3)

vao = VAO(pos_ssbo)


glPointSize(5)  #not working on rpi
glClearColor(0.2,0.2,0.2,1)

@window.event
def on_draw():
    # window.clear()
    glClear(GL_COLOR_BUFFER_BIT)

    sha_posspd.execute(pos_ssbo.size, Pos=pos_ssbo,Spd=spd_ssbo)
    
    sha.use()
    vao.draw()
    # sha.draw(pos_ssbo.size//3, position=pos_ssbo)
    
    time.sleep(0.01)

pyglet.app.run()

# while not glfw.window_should_close(window):    
#     glClear(GL_COLOR_BUFFER_BIT)
    
#     glDrawArrays(GL_POINTS, 0, 3)

#     glfw.poll_events()
#     glfw.swap_buffers(window)
