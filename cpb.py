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
        
        #it works!
        glTexStorage2D = _gl.glTexStorage2D
        glTexStorage3D = _gl.glTexStorage3D
        glBindImageTexture = _gl.glBindImageTexture
        
        
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
print('glBindImageTexture',bool(glBindImageTexture))
print('glProgramUniformMatrix4fv',bool(glProgramUniformMatrix4fv))
print("GL_MAX_COMPUTE_SHARED_MEMORY_SIZE", glGetIntegerv(GL_MAX_COMPUTE_SHARED_MEMORY_SIZE))#32kbrpi
print("GL_MAX_COMPUTE_WORK_GROUP_SIZE", glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE,0))#256rpi
print('GL_MAX_ARRAY_TEXTURE_LAYERS', glGetIntegerv(GL_MAX_ARRAY_TEXTURE_LAYERS))



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



#texture uses 2d cache. Morton layout (Z-order curve)
#NV 64bytes 8x8 2D tile | arm 16x16 or 32.


#texture types for GLSL:
# samplerBuffer for TBO
# Image2D 2D layout, texture cache. imageLoad/imageStore
# sampler2D ReadOnly. mipmap,filter,warp,normalized coords. great for 2D RO?

# sampler2D / sampler2DArray / isampler2D usampler2D

#to get data from texture-likes
# fragColor = texture(tex, vec3(fs_uv,Index));
# texelFetch gets data with int texel coords.

#uniform samplerBuffer myTBO;
#float v = texelFetch(myTBO, index).r;


# layout(binding=0) uniform sampler2DArray texArr;
# vec4 c = texture(texArr, vec3(uv, layer_index));
# layout(rgba8, binding=0) uniform readonly image2DArray img;
# vec4 v = imageLoad(img, ivec3(x, y, layer_index));

class Texture:
    "for image layer"
    def __init__(self, npimg):
        height,width = npimg.shape

        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        # glBindImageTexture #for compute

        # glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width, height, 0, GL_RED, GL_FLOAT, npimg)
        # glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGB, GL_FLOAT, npimg[::-1])        
        
        #more static way..
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32F, width,height)
        glTexSubImage2D(GL_TEXTURE_2D,
                0,        # mip level
                0, 0,   # xoffset, yoffset
                width, height,     # update region size
                GL_RED,
                GL_FLOAT,
                npimg)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)#GL_NEAREST GL_LINEAR GL_LINEAR_MIPMAP_LINEAR
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        # glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0)
        # glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0)
        # glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        # glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        self.ID = texture

    def bind(self):
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.ID)

npimg = np.arange(8*8).reshape(8,8).astype(np.float32)/8**2
tex = Texture(npimg)

class TextureArray:
    "for image layer"
    def __init__(self, npimgs):

        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D_ARRAY, texture)
        
        layer_count = len(npimgs)
        height,width = npimgs[0].shape        
        # print(bool(glTexStorage3D))
        glTexStorage3D(GL_TEXTURE_2D_ARRAY,
                       1,                 # mipmap levels
                       GL_R32F,          # internal format
                       width,
                       height,
                       layer_count)

        for i, img in enumerate(npimgs):
            glTexSubImage3D(GL_TEXTURE_2D_ARRAY,
                            0, 0, 0, i,
                            width, height, 1,
                            GL_RED, GL_FLOAT,
                            img[::-1])
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        self.ID = texture
        self.layer_count = layer_count
        # self.format = GL_R32F

    def bind(self):
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D_ARRAY, self.ID)
    def bind_compute(self):
        glBindImageTexture(0, self.ID, 0, GL_TRUE, 0, GL_READ_WRITE, GL_R32F)
        #GL_TRUE for layered→ array all layers..?



#compute
#layout(r32f, binding = 0) uniform image2DArray img;
#glBindImageTexture(0, texture, 0, GL_TRUE, 0, GL_READ_WRITE, GL_R32F);#GL_TRUE는 layered→ array의 모든 레이어 접근 가능.




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
            # print(glGetProgramResourceiv(compute_posspd.program, GL_SHADER_STORAGE_BLOCK, 0, 1,[GL_BUFFER_BINDING], 1))
            
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

    def set_uniform(self, name,value, vtype='f'):
        loc = self.get_loc_uniform(name)
        self.use()#for future, it will be removed. only for RPI,legacy.
        if vtype == 'f':
            glUniform1f(loc, value)  #set variable
        elif vtype == 'i':
            glUniform1i(loc, value)  #set variable
        elif vtype == 'ui':
            glUniform1ui(loc, value)  #set variable
        # glProgramUniform1f / DSA

    def set_uniform3(self, name,value, vtype='f'):
        loc = self.get_loc_uniform(name)
        self.use()#for future, it will be removed. only for RPI,legacy.
        if vtype == 'f':
            glUniform3fv(loc, 1,value)  #set variable
        elif vtype == 'i':
            glUniform3iv(loc, 1,value)  #set variable
        elif vtype == 'ui':
            glUniform3uiv(loc, 1,value)  #set variable
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

    def dispatch(self,X,Y=1,Z=1):
        # "make sure to use()!"
        self.use()#made it sure.
        if Y==Z==1:
            N = X
            dispatch_size = math.ceil(N/BLOCKSIZE)
            glDispatchCompute(dispatch_size,1,1)
        else:
            glDispatchCompute(X,Y,Z)
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























#======================================================
#======================================================
#======================================================
#======================================================
#===============FORGET THE CLASSES AVOBE===============
#===============FORGET THE CLASSES AVOBE===============
#===============FORGET THE CLASSES AVOBE===============
#===============FORGET THE CLASSES AVOBE===============
#======================================================
#======================================================
#======================================================
#======================================================

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
compute_posspd = ComputeShader(compute_src_posspd)
compute_posspd.set_uniform("dt",0.1)



pos = np.arange(10, dtype=np.float32)  # 0~15
spd = np.ones(10, dtype=np.float32)
pos_ssbo = SSBO(pos)
spd_ssbo = SSBO(spd)

# compute_posspd.bind_ssbo("Pos",pos_ssbo)
# compute_posspd.bind_ssbo("Spd",spd_ssbo)
# compute_posspd.dispatch(pos_ssbo.size)
# or in short,
compute_posspd.execute(pos_ssbo.size, Pos=pos_ssbo,Spd=spd_ssbo)

print(pos_ssbo.read())
# exit()


#============part 2. 3d particles.
N = 10000
pos = np.zeros(N*3, dtype=np.float32)
spd = np.random.rand(N*3).astype(np.float32)-0.5
pos_ssbo = SSBO(pos)
spd_ssbo = SSBO(spd)

compute_posspd.execute(pos_ssbo.size, Pos=pos_ssbo,Spd=spd_ssbo)









#==============================cnn operations







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

vert_point = """
#version 310 es
precision highp float;

uniform mat4 ProjectionView;

layout(location = 0) in vec3 position;

void main() {
    gl_Position = ProjectionView * vec4(position, 1.0);
}
"""
frag_point = """
#version 310 es
precision highp float;

out vec4 color;

void main(void){
    color = vec4(1,0,1, 1.0);
}
"""


vert_tex = """
#version 310 es
precision highp float;

uniform mat4 ProjectionView;
uniform vec3 Coords;
uniform float AddSize;

const vec3 rect[6] = vec3[6](
    vec3(0,0,0),
    vec3(1,0,0),
    vec3(1,1,0),
    
    vec3(0,0,0),
    vec3(1,1,0),
    vec3(0,1,0)
);

out vec2 uv_out;
out flat int rect_idx;

void main() {
    int pos_idx = gl_VertexID % 6; // 0,1,2,3,4,5, 0,1,2,3,4,5,
    rect_idx = gl_VertexID / 6; // 0,0,0,0,0,0, 1,1,1,1,1,1,
    
    vec3 position = rect[pos_idx];
    uv_out = position.xy;
    
    position *= (1.0+AddSize);//implicit convert error if 1+addsize.(not float)
    position.z += -float(rect_idx)*0.5;
    position += Coords;
    gl_Position = ProjectionView * vec4(position, 1.0);
}
"""
frag_tex = """
#version 310 es
precision highp float;
precision highp sampler2DArray;


flat in int rect_idx;
in vec2 uv_out;
out vec4 color;

//uniform uint tidx;
//uniform sampler2D tex;
uniform sampler2DArray texA;

vec3 jet(float x) {
    // x: 0.0 ~ 1.0
    float r = clamp(1.5 - abs(4.0 * x - 3.0), 0.0, 1.0);
    float g = clamp(1.5 - abs(4.0 * x - 2.0), 0.0, 1.0);
    float b = clamp(1.5 - abs(4.0 * x - 1.0), 0.0, 1.0);
    return vec3(r, g, b);
}

void main(void){
    //color = vec4(uv_out,0, 1.0);
    //color = texture(tex, uv_out);
    
    //ivec2 size = textureSize(tex, 0);
    //ivec2 coord = ivec2(uv_out * vec2(size));
    //color = texelFetch(tex, coord, 0);

    color = texture(texA, vec3(uv_out, rect_idx));
    color.yz = color.xx;
    //if (color.x<0.001){
    //    color.a = 0.0;
    //}
    //color = vec4(jet(float(color.x)), 1.0);

    //texel requires uint indexing! not [0-1] of uv.
    //For simplicity, we draw using texture().
    //color = texelFetch(texA, ivec3(uv_out, rect_idx), 0);
}
"""
sha_point = DrawShader(vs=vert_point,fs=frag_point)
sha_rect = DrawShader(vs=vert_tex,fs=frag_tex)


projection = matrix44.create_perspective_projection(45,16/9,0.01,100)#fov ratio near far
view = matrix44.create_look_at([5,5,5], [0,0,0], [0,1,0]) # view target up
projection_view = (projection.T@view.T).T #to pretty print(and gl-preffered), pyrr is col.major.

sha_point.set_uniform4x4("ProjectionView", projection_view)
sha_rect.set_uniform4x4("ProjectionView", projection_view)






# Input channel (RGB, 3)
# Output channel = feature map / filter count.
# feature depth/feature dimension
# kernel has depth, kernel tensor has shape = (32,1,3,3) , depth=32.
class Conv2d:
    def __init__(self, weight, outsize):
        out_ch,in_ch,kh,kw = weight.shape
        #(32, 1, 3, 3) for 3x3 input:1, output:32.
        self.kw = kw
        self.kh = kh
        
        self.in_ch = in_ch
        self.out_ch = out_ch

        #for convinience, but confusing naming!
        # self.size = kw
        # self.depth = out_ch

        # for i in range(in_ch):
        #keep the shape to 3d spartial layout.
        self.tex_arrs = []
        for kernels in weight:
            npimgs = [k for k in kernels]
            tex_arr = TextureArray(npimgs)
            self.tex_arrs.append(tex_arr)

        self.weight = weight
        #we need output layers, too.-> not here! unknown to input.
        #those will be traced by the class mimicing nn.Modules..
        #..that traces the graph.
        #when x,input given, layer's output will be deterministic during runtime.
        #but keep those lines here, to keep the complexity.
        self.tex_outputs = []

        npimg = np.zeros((outsize,outsize),dtype=np.float32)
        npimgs = [npimg for i in range(out_ch)]
        for depth in range(in_ch):
            tex_arr = TextureArray(npimgs)
            self.tex_outputs.append(tex_arr)


# gl_WorkGroupID dispatch group coords
# gl_LocalInvocationID local sized
# gl_GlobalInvocationID global sized
# gl_WorkGroupSize local sizes
# gl_NumWorkGroups dispatch dispatch sizes
# gl_LocalInvocationIndex flatten! easy to access ssbo.
compute_src_conv2d ="""
#version 310 es
precision highp float;
precision highp sampler2DArray;
precision highp image2DArray;

uniform float dt;

layout(local_size_x = 16, local_size_y = 16) in;

layout(r32f, binding = 0) uniform writeonly image2DArray img;

void main() {

    ivec3 coord = ivec3(gl_GlobalInvocationID.xy, 0); // layer=0
    float val = float(gl_LocalInvocationID.x)/32.0 + float(gl_LocalInvocationID.y)/32.0;
    imageStore(img, coord, vec4(val, 0.0, 0.0, 0.0)); // R
}
"""
compute_conv2d = ComputeShader(compute_src_conv2d)
# compute_conv2d.set_uniform("dt",0.1)





data = np.load("params.npz")
# for k in data.files:
#     arr = data[k]
#     print(k, arr.shape, arr.dtype)
# conv1_weight (32, 1, 3, 3) float32
# conv1_bias (32,) float32
# conv2_weight (64, 32, 3, 3) float32
# conv2_bias (64,) float32
# fc1_weight (128, 3136) float32
# fc1_bias (128,) float32
# fc2_weight (10, 128) float32
# fc2_bias (10,) float32

conv1_weight = data['conv1_weight']
# print(conv1_weight.shape)
# print(conv1_weight)

conv2d = Conv2d(conv1_weight, outsize=28)

conv2d.tex_outputs[0].bind_compute()
compute_conv2d.dispatch(2,2,1)

test_data = np.load("fmnist_test_normalized.npz")
imgs = test_data["images"]        # (N,1,28,28)
# imgs = (imgs+1)/2
labels = test_data["labels"]
# print(imgs.shape)#(10000, 1, 28, 28)


# npimg = np.arange(8*8).reshape(8,8).astype(np.float32)/8**2
XY = 3
npimgs = [np.random.rand(XY*XY).reshape(XY,XY).astype(np.float32) for i in range(8)]
tex_arr = TextureArray(npimgs)

XY = 7
npimgs = [np.random.rand(XY*XY).reshape(XY,XY).astype(np.float32) for i in range(4)]
npimgs = [imgs[0].reshape(28,28)]
tex_arr2 = TextureArray(npimgs)




vao_point = glGenVertexArrays(1)
glBindVertexArray(vao_point)
glBindBuffer(GL_ARRAY_BUFFER, pos_ssbo.ID)
glEnableVertexAttribArray(0)
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)#index size type normalize stride offsetptr

vao_rect = glGenVertexArrays(1)


glPointSize(5)  #not working on rpi
glClearColor(0.0,0.2,0.2,1)
glEnable(GL_DEPTH_TEST)
glEnable(GL_BLEND)
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
@window.event
def on_draw():
    # window.clear()
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

    compute_posspd.execute(pos_ssbo.size, Pos=pos_ssbo,Spd=spd_ssbo)
    
    # sha_point.use()
    # glBindVertexArray(vao_point)
    # glDrawArrays(GL_POINTS, 0, N)

    sha_rect.use()
    glBindVertexArray(vao_rect)
    sha_rect.set_uniform('AddSize',0)
    
    # sha_rect.set_uniform3('Coords',(-2,0,1))
    # tex_arr.bind()
    # glDrawArrays(GL_TRIANGLES, 0, 6*tex_arr.layer_count)#6=2 triangles.
    
    for i,tex_arr in enumerate(conv2d.tex_arrs):
        # sha_rect.set_uniform3('Coords',(-2, -i, 0))
        sha_rect.set_uniform3('Coords',(-2, 0, 2-i*0.5))
        tex_arr.bind()        
        glDrawArrays(GL_TRIANGLES, 0, 6*tex_arr.layer_count)#6=2 triangles.

    sha_rect.set_uniform('AddSize',1)
    for i,tex_arr in enumerate(conv2d.tex_outputs):
        # sha_rect.set_uniform3('Coords',(-2, -i, 0))
        sha_rect.set_uniform3('Coords',(1.5, 0, 2-i*0.5))
        tex_arr.bind()
        glDrawArrays(GL_TRIANGLES, 0, 6*tex_arr.layer_count)#6=2 triangles.

    sha_rect.set_uniform3('Coords',(-7,0,0))
    sha_rect.set_uniform('AddSize',2)
    tex_arr2.bind()
    glDrawArrays(GL_TRIANGLES, 0, 6*tex_arr2.layer_count)#6=2 triangles.

    time.sleep(0.01)

pyglet.app.run()

# while not glfw.window_should_close(window):    
#     glClear(GL_COLOR_BUFFER_BIT)
    
#     glDrawArrays(GL_POINTS, 0, 3)

#     glfw.poll_events()
#     glfw.swap_buffers(window)
