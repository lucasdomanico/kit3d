import { PicoGL } from './lib.picogl/picogl.js' //# PicoGL
import shader from './shader.js'

let shader_global_prefix = 'x_'

let nil = null //# null

let webgl = (canvas) => {
    console.log('::::', canvas)
    let app = PicoGL.createApp(canvas, { //# PicoGL createApp
        alpha: true,                     //# alpha
        antialias: true,                 //# antialias
        depth: false,                    //# depth
        stencil: false,                  //# stencil
        premultipliedAlpha: true,        //# premultipliedAlpha
        // preserveDrawingBuffer: true,
        // desynchronized: true // only for canvas_element, ignored by offscreen
        // powerPreference: "high-performance"
    })
    let texoptions = {
        premultiplyAlpha: true,      //# premultiplyAlpha
        flipY: true,                 //# flipY
        wrapS: PicoGL.CLAMP_TO_EDGE, //# wrapS PicoGL CLAMP_TO_EDGE
        wrapT: PicoGL.CLAMP_TO_EDGE, //# wrapT PicoGL CLAMP_TO_EDGE
        // wrapS: PicoGL.REPEAT, //# wrapS PicoGL CLAMP_TO_EDGE
        // wrapT: PicoGL.REPEAT, //# wrapT PicoGL CLAMP_TO_EDGE
        // WEBGL_INFO.MAX_TEXTURE_ANISOTROPY
        maxAnisotropy: PicoGL.WEBGL_INFO.MAX_TEXTURE_ANISOTROPY,
    }
    // program -> geometries -> drawcall
    let drawcalls = new Map() //# Map
    let wrap_fbo = (fbo, colors, depth) => ({ fbo, colors, depth })
    let tmpbuf = wrap_fbo(app.createFramebuffer(), nil, nil) //# createFramebuffer
    let createTexture2D_hack = (pixels, w, h, opt, $clone) => {
        let tex = app.createTexture2D(pixels, w, h, opt) //# createTexture2D
        let clone = $clone?? (() => createTexture2D_hack(nil, w, h, opt, clone))
        return { tex, clone }
    }
    return {
        resize: (w, h) => {
            app.resize(w, h)
        },
        width: () => {
            return app.width //# width
        },
        height: () => {
            return app.height //# height
        },
        varray: ([indices, positions, uvs, normals, tangents, joints, weights]) => {
            // console.log(uvs)
            // ver uvs de tokyo.bake
            // for(let i = 0; i < uvs.length; i+= 2) {
            //     uvs[i] += 2
            // }
            let attr = (vao, n, bufargs, opt) => {
                let buf = app.createVertexBuffer(...bufargs)  //# createVertexBuffer
                return vao.vertexAttributeBuffer(n, buf, opt) //# vertexAttributeBuffer
            }
            let u = PicoGL.UNSIGNED_INT //# PicoGL UNSIGNED_INT
            let f = PicoGL.FLOAT        //# PicoGL FLOAT
            let vao = app.createVertexArray() //# createVertexArray
            vao.indexBuffer(app.createIndexBuffer(u, 3, indices)) //# indexBuffer createIndexBuffer
            attr(vao, 0, [f, 3, positions])
            attr(vao, 1, [f, 2, uvs])
            attr(vao, 2, [f, 3, normals])
            attr(vao, 3, [f, 4, tangents])
            attr(vao, 4, [u, 4, joints], { integer:true }) //# integer
            attr(vao, 5, [f, 4, weights])
            return vao
        },
        program: (vertex, fragment) => {
            return app.createProgram(vertex, fragment) //# createProgram
        },
        isvarray: (o) => o.constructor.name === 'VertexArray',           //# constructor name
        istexture: (o) => o.tex && o.tex.constructor.name === 'Texture', //# constructor name
        image: (pixels, w, h, isdata, flip, wraps, wrapt) => {
            let wrap = (s) => {
                if(s === 'clamp')  return PicoGL.CLAMP_TO_EDGE
                if(s === 'repeat') return PicoGL.REPEAT
                throw 'wrap ' + s
            }
            let opt = isdata? {
                // no mipmaps (and no flip, no premulalpha)
                minFilter: PicoGL.NEAREST, //# minFilter PicoGL NEAREST
                magFilter: PicoGL.NEAREST  //# magFilter PicoGL NEAREST
            } : { ...texoptions, flipY:flip, wrapS:wrap(wraps), wrapT:wrap(wrapt) } //# flipY
            // not needed in new PicoGL
            if(pixels.constructor === Float32Array) { //# constructor Float32Array
                opt.internalFormat = PicoGL.RGBA32F //# internalFormat PicoGL RGBA32F
            }
            // console.log('g3d', w, h)
            return createTexture2D_hack(pixels, w, h, opt, nil)
        },
        clone_texture: (t) => {
            return t.clone()
        },
        texture_width: (t) => t.tex.width,   //# width
        texture_height: (t) => t.tex.height, //# height
        texture_delete: (t) => {
            t.tex.delete() //# delete
            // t.clone = nil
        },
        set_data: (texture, pixels) => {
            texture.tex.data(pixels) //# data
        },
        copy_start: (write) => {
            tmpbuf.fbo.resize(write.tex.width, write.tex.height) //# resize width height
            tmpbuf.fbo.colorTarget(0, write.tex) //# colorTarget
        },
        // only use this buffer for clear and draw operations
        copy_buffer: () => {
            return tmpbuf
        },
        copy_end: () => {
            // tmpbuf.deattach
        },
        // read: (tex, pixels, flip) => {
        //     let n = 0
        //     tmpbuf.fbo.resize(tex.width, tex.height)
        //     tmpbuf.fbo.colorTarget(n, tex)
        //     app.readFramebuffer(tmpbuf.fbo)
        //     // else app.defaultReadFramebuffer()
        //     // tex.bind(Math.max(tex.currentUnit, 0)) // gl flip readpixels doesnt works... ¿?
        //     // app.gl.pixelStorei(app.gl.UNPACK_FLIP_Y_WEBGL, false)
        //     // console.log(app.gl.getParameter(app.gl.UNPACK_FLIP_Y_WEBGL))
        //     app.gl.readBuffer(app.gl.COLOR_ATTACHMENT0 + n)
        //     app.gl.readPixels(0, 0, tex.width, tex.height, app.gl.RGBA, app.gl.UNSIGNED_BYTE, pixels)
        //     if(flip) {
        //         // wmatch.js all .read()
        //     }
        // },
        buffer: (w, h, colors, depth, msaa) => {
            // console.log(msaa)
            let fbo = app.createFramebuffer() //# createFramebuffer
            let color_textures = []
            let depth_textures = []
            let attach = (textures, texture) => {
                textures.push(texture) //# push
                return texture.tex
            }
            let renderbuf = (t) => {
                let max_samples = PicoGL.WEBGL_INFO.SAMPLES //# PicoGL WEBGL_INFO SAMPLES
                return app.createRenderbuffer(w, h, t, max_samples) //# createRenderbuffer
            }
            for(let i = 0; i < colors; i++) {
                let target = msaa?
                    renderbuf(PicoGL.RGBA8) : //# PicoGL RGBA8
                    attach(color_textures, createTexture2D_hack(nil, w, h, texoptions, nil))
                fbo.colorTarget(i, target) //# colorTarget
            }
            if(depth) {
                let format = PicoGL.DEPTH_COMPONENT32F //# PicoGL DEPTH_COMPONENT32F
                let target = msaa?
                    renderbuf(format) :
                    attach(depth_textures, createTexture2D_hack(nil, w, h, {
                        internalFormat: format //# internalFormat
                    }, nil))
                fbo.depthTarget(target) //# depthTarget
            }
            return wrap_fbo(fbo, color_textures, depth_textures.length == 0? nil : depth_textures[0]) //# length
        },
        // color and depth, always for non-msaa! (if so, this returns a render-attachment, not texture)
        color: (buffer, n) => {
            if(!n) n = 0
            return buffer.colors[n]
            return buffer.fbo.colorAttachments[n]
        },
        depth: (buffer) => {
            return buffer.depth
            return buffer.fbo.depthAttachment
        },
        blit_framebuffers_2: (draw, read, color, depth) => {
            if(draw) app.drawFramebuffer(draw.fbo) //# drawFramebuffer
            else app.defaultDrawFramebuffer()      //# defaultDrawFramebuffer
            if(read) app.readFramebuffer(read.fbo) //# readFramebuffer
            else app.defaultReadFramebuffer()      //# defaultReadFramebuffer
            let mask = 0
            if(color) mask |= app.gl.COLOR_BUFFER_BIT //# gl COLOR_BUFFER_BIT
            if(depth) mask |= app.gl.DEPTH_BUFFER_BIT //# gl DEPTH_BUFFER_BIT
            app.depthMask(true)       //# depthMask
            app.blitFramebuffer(mask) //# blitFramebuffer
        },
        delete: (buffer) => {
            buffer.fbo.delete() //# delete
        },
        clear: (buffer, color, depth) => {
            let bits = app.gl.STENCIL_BUFFER_BIT      //# gl STENCIL_BUFFER_BIT
            if(color) bits |= app.gl.COLOR_BUFFER_BIT //# gl COLOR_BUFFER_BIT
            if(depth) bits |= app.gl.DEPTH_BUFFER_BIT //# gl DEPTH_BUFFER_BIT
            app.clearBits = bits                      //# clearBits
            //      app.clearBits = app.gl.COLOR_BUFFER_BIT | app.gl.DEPTH_BUFFER_BIT | app.gl.STENCIL_BUFFER_BIT
            //      clearMask(app.gl.COLOR_BUFFER_BIT | app.gl.DEPTH_BUFFER_BIT | app.gl.STENCIL_BUFFER_BIT)
            //      if mask isn't true opengl doesn't clears the depth buffer
            app.depthMask(true) //# depthMask
            // app.noDepthTest()
            if(buffer) app.drawFramebuffer(buffer.fbo) //# drawFramebuffer
            else app.defaultDrawFramebuffer()          //# defaultDrawFramebuffer
            app.clearColor(0, 0, 0, 0).clear()         //# clearColor clear
        },
        draw: (buffer, varray, program, args, instances, depth, mask, blend, backside, w, h) => {
            let programs = mapget(drawcalls, program, () => new Map()) //# Map
            let drawcall = mapget(programs, varray, () => app.createDrawCall(program, varray)) //# createDrawCall
            // console.log(varray.instanced, varray.numInstances)
            // varray.instanced = true
            // ************ ask instances API
            varray.numInstances = instances      //# numInstances
            drawcall.numInstances[0] = instances //# numInstances
            let uniformv = Object.keys(program.uniforms) //# Object keys uniforms
            let samplerv = Object.keys(program.samplers) //# Object keys samplers
            let has = (k) => uniformv.includes(k) || samplerv.includes(k) //# includes
            args.forEach(([k, t, v]) => { //# forEach
                if(t === 'sampler2D') {
                    drawcall.texture(shader_global_prefix + k, v.tex) //# texture
                    return
                }
                let array = t.endsWith(']') ? '[0]' : '' //# endsWith
                let key = shader_global_prefix + k + array
                if(has(key)) {
                    // console.log(key, v)
                    drawcall.uniform(key, v) //# uniform
                    return
                }
                return
            })
            app.viewport(0, 0, w, h) //# viewport
            // console.log('maskdepth', mask, depth)
            depth? app.depthTest() : app.noDepthTest() //# depthTest noDepthTest
            app.depthFunc(PicoGL.LEQUAL) //# depthFunc PicoGL LEQUAL
            // depth write
            app.depthMask(mask) //# depthMask
            let one =                 PicoGL.ONE                 //# PicoGL ONE
            let one_minus_src_alpha = PicoGL.ONE_MINUS_SRC_ALPHA //# PicoGL ONE_MINUS_SRC_ALPHA
            let dst_color =           PicoGL.DST_COLOR           //# PicoGL DST_COLOR
            let one_minus_src_color = PicoGL.ONE_MINUS_SRC_COLOR //# PicoGL ONE_MINUS_SRC_COLOR
            let blends = new Map([ //# Map
                ['normal',   [one,       one_minus_src_alpha, one, one_minus_src_alpha]],
                ['add',      [one,       one,                 one, one]],
                ['multiply', [dst_color, one_minus_src_alpha, one, one_minus_src_alpha]],
                ['screen',   [one,       one_minus_src_color, one, one_minus_src_alpha]]
            ])
            // ['normal',   [PicoGL.ONE,       PicoGL.ONE_MINUS_SRC_ALPHA, PicoGL.ONE, PicoGL.ONE_MINUS_SRC_ALPHA]],
            // ['add',      [PicoGL.ONE,       PicoGL.ONE,                 PicoGL.ONE, PicoGL.ONE]],
            // ['multiply', [PicoGL.DST_COLOR, PicoGL.ONE_MINUS_SRC_ALPHA, PicoGL.ONE, PicoGL.ONE_MINUS_SRC_ALPHA]],
            // ['screen',   [PicoGL.ONE,       PicoGL.ONE_MINUS_SRC_COLOR, PicoGL.ONE, PicoGL.ONE_MINUS_SRC_ALPHA]]
            let $blend = blends.get(blend? blend : 'normal') //# get
            app.blend() //# blend
            app.blendFuncSeparate($blend[0], $blend[1], $blend[2], $blend[3]) //# blendFuncSeparate
            if(backside) app.drawBackfaces() //# drawBackfaces
            else app.cullBackfaces()         //# cullBackfaces
            // app.noScissorTest()
            // app.noStencilTest()
            // app.noRasterize()
            // console.log(PicoGL.WEBGL_INFO)
            // PicoGL.WEBGL_INFO.MULTI_DRAW_INSTANCED = false
            if(buffer) app.drawFramebuffer(buffer.fbo) //# drawFramebuffer
            else app.defaultDrawFramebuffer() //# defaultDrawFramebuffer
            drawcall.draw() //# draw
        },
        delete: () => {}
    }
}

let mapget = (map, k, init) => {
    ///////////////////////////////////////////////////
    // return map.get(k)?? init()
    ///////////////////////////////////////////////////
    if(!map.has(k)) map.set(k, init()) //# has set
    return map.get(k) //# get
}

let def = (value, init) => value === undefined? init : value


let u32 = Uint32Array  //# Uint32Array
let f32 = Float32Array //# Float32Array
let quad_mesh = [
    new u32([0, 2, 1, 2, 3, 1]),
    new f32([-0.5, 0.5, 0, 0.5, 0.5, 0, -0.5, -0.5, 0, 0.5, -0.5, 0]),
    // new Float32Array([-1, 1, 0, 1, 1, 0, -1, -1, 0, 1, -1, 0]),
    new f32([0, 1, 1, 1, 0, 0, 1, 0]),
    new f32([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]),
    new f32([1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1]),
    new u32([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    new f32([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
]

let mesh_wrap = (varray) => ({ _:varray })

let g3d = (canvas) => {
    let gl = webgl(canvas)
    let quad = mesh_wrap(gl.varray(quad_mesh))
    let programs = new Map() //# Map
    let varray = (mesh) => {
        if(!mesh) return quad._
        if(gl.isvarray(mesh._)) {
            return mesh._
        }
        throw mesh
    }
    let program = (code, args) => {
        let types = args.map(([k, t]) => [k, t]) //# map
        let hash = types.flat().join(',')        //# flat join
        return mapget(programs, hash + code, () => {
            let [v, p] = shader(code, types, shader_global_prefix)
            return gl.program(v, p)
        })
    }
    let is_texture = (value) => value._ && gl.istexture(value._)
    let args = (aspect, instances, uniforms) => {
        if(!uniforms) uniforms = []
        let v = [
            ['aspect',    'float', aspect],
            ['instances', 'int',   instances]
        ]
        Object.keys(uniforms).forEach((key) => { //# Object keys forEach
            let value = uniforms[key]
            let type = ''
            if(is_texture(value)) {
                type = 'sampler2D'
                value = value._
            }
            else if(typeof value === 'number') {
                type = 'float'
            }
            else if(Array.isArray(value) && value.length === 2) { //# Array isArray length
                type = value[0]
                value = value[1]
            }
            else throw key
            v.push([key, type, value]) //# push
        })
        return v
    }
    let solvezbuf = (zbuf, depth, mask) => {
        depth ??= false
        mask ??= true
        if(!zbuf && depth) throw ''
        return [depth, depth? mask : false] // second expr should be done here???
    }
    let draw = (buffer, w, h, zbuf, { mesh, shader, uniforms, instances, depth, mask, blend, backside, clear }) => {
        //  if(color === false) gl.colorMask(false, false, false, false)
        if(clear) gl.clear(buffer, true, true)
        if(!shader) throw 'shader not given'
        let $instances = def(instances, 1)
        let $args = args(w / h, $instances, uniforms)
        let $varray = varray(mesh)
        let $program = program(shader, $args)
        let [$depth, $mask] =  solvezbuf(zbuf, depth, mask)
        let $blend = def(blend, '')
        let $backside = def(backside, false)
        gl.draw(buffer, $varray, $program, $args, $instances, $depth, $mask, $blend, $backside, w, h)
    }
    let clear = (buf, color, depth) => {
        let $color = def(color, true)
        let $depth = def(depth, true)
        gl.clear(buf, $color, $depth)
    }
    let copy = (write, read) => {
        gl.copy_start(write)
        draw(gl.copy_buffer(), gl.texture_width(write), gl.texture_height(write), false, {
            uniforms: { read },
            shader: `vec4 pixel(px p) { return texture(p.read, p.uv); }`
        })
        gl.copy_end()
    }
    let texture_wrap = (texture, flip) => ({
        _: texture,
        width: () => gl.texture_width(texture),
        height: () => gl.texture_height(texture),
        set: (pixels) => {
            if(is_texture(pixels)) return copy(texture, pixels)
            gl.set_data(texture, pixels)
        },
        // read: (pixels) => {
        //     if(!pixels) {
        //         pixels = new Uint8ClampedArray(texture.width * texture.height * 4)
        //     }
        //     gl.read(texture, pixels, flip)
        //     return pixels
        // },
        delete: () => { gl.texture_delete(texture) }
    })
    let none = texture_wrap(gl.image(new Float32Array(4), 1, 1, false, false, 'clamp', 'clamp'), false) //# Float32Array
    let pixsize = (pixels, w, h) => {
        if(w && h) return [pixels, w, h]
        if(pixels.constructor === HTMLVideoElement) { //# constructor HTMLVideoElement
            return [pixels, pixels.videoWidth, pixels.videoHeight] //# videoWidth videoHeight
        }
        if(pixels.constructor === Float32Array) { //# constructor Float32Array
            let size = Math.ceil(Math.sqrt(pixels.length / 4)) //# Math ceil sqrt length
            let v = new Float32Array(size * size * 4) //# Float32Array
            v.set(pixels, 0) //# set
            return [v, size, size]
        }
        if(pixels.width && pixels.height) return [pixels, pixels.width, pixels.height] //# width height
        throw pixels
    }
    return {
        resize: (w, h) => gl.resize(w, h),
        width: () => gl.width(),
        height: () => gl.height(),
        flush: (buf) => {
            draw(nil, gl.width(), gl.height(), false, {
                clear: true,
                uniforms: { buf:buf.color() },
                shader: `
                    vec4 pixel(px p) {
                        return texture(p.buf, p.uv);
                    }
                `
            })
        },
        quad: () => quad,
        blank: () => none,
        mesh: (obj) => mesh_wrap(gl.varray(obj)),
        texture: (pixels, args) => {

            if(is_texture(pixels)) {
                let t = gl.clone_texture(pixels._)
                copy(t, pixels)
                return texture_wrap(t, t.tex.flipY) //# flipY
            }

            let isdata = (pixels.constructor === Float32Array) //# constructor Float32Array
            let flip = def(args?.flip, isdata? false : true)
            let wraps = def(args?.wraps, 'clamp')
            let wrapt = def(args?.wrapt, 'clamp')
            let [$pixels, w, h] = pixsize(pixels, args?.width, args?.height)
            return texture_wrap(gl.image($pixels, w, h, isdata, flip, wraps, wrapt), flip)
        },        
        buffer: (args) => {
            let $colors = def(args?.colors, 1)
            let $depth =  def(args?.depth, false)
            let w = def(args?.width, gl.width())
            let h = def(args?.height, gl.height())
            let $msaa = def(args?.msaa, false)
            let buf = gl.buffer(w, h, $colors, $depth, false)
            gl.clear(buf, true, true) // depth texture may be not cleared
            let msaa = $msaa? gl.buffer(w, h, $colors, $depth, true) : nil
            let tdepth = $depth? texture_wrap(gl.depth(buf), false) : nil
            let tcolors = [...Array($colors).keys()].map((n) => { //# Array keys map
                return texture_wrap(gl.color(buf, n), false)
            })
            let sync = true
            let dosync = () => {
                if(sync) return
                if(msaa) {
                    gl.blit_framebuffers_2(buf, msaa, true, true)
                }
                sync = true
            }
            return {
                _: { buf, msaa },
                width: () => w,
                height: () => h,
                clear: (color, depth) => {
                    if(msaa) clear(msaa, color, depth)
                    clear(buf, color, depth)
                    sync = true
                },
                draw: (data) => {
                    let target = msaa? msaa : buf
                    draw(target, w, h, $depth, data)
                    sync = false
                },
                blit: (read, color, depth) => {
                    color ??= true
                    depth ??= true
                    let target = msaa? msaa : buf
                    let r = read._.msaa?? read._.buf
                    gl.blit_framebuffers_2(target, r, color, depth)
                    sync = false
                },
                color: (n=0) => {
                    dosync()
                    return tcolors[n]
                },
                depth: () => {
                    dosync()
                    return tdepth
                }
            }
        }
    }
}

/*

    

*/



export default g3d