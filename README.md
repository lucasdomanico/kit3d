# kit3d
Javascript API for doing computer graphics, providing a more convenient interface for immediate-mode rendering.

Web: https://github.com/lucasdomanico/kit3d

```typescript
    interface kit3d {
        new(c:HTMLCanvasElement):g3d
    }
    interface g3d {
        mesh(md:meshdata):mesh
        buffer(options?:{ width?:int, height?:int, depth?:boolean, msaa?:boolean, colors?:int }):buffer
        texture(p:pixels, options?:{ width?:int, height?:int, flip?:boolean, wraps:string, wrapt:string }):texture
        flush(b:buffer):void
    }
    interface mesh {} // nominal
    interface buffer {
        width():int
        height():int
        draw(dc:drawcall):void
        blit(read:buffer, color?:boolean, depth?:boolean):void
        clear(color?:boolean, depth?:boolean):void
        color(n?:int):texture
        depth():texture
    }
    interface texture {
        width():int
        height():int
        set(p:pixels):void
        get(p:Uint8Array | Float32Array):void
    }
    interface drawcall {
        mesh      ?:mesh        // quad
        shader    :string       // required
        uniforms  ?:{ [id:string]:uniform }
        instances ?:int         // 1
        depth     ?:boolean     // false (depth test)
        mask      ?:boolean     // true  (depth write)
        blend     ?:string      // 'normal'
        backside  ?:boolean     // false
        clear     ?:boolean     // false
    }
    type uniform = texture | number | [type:string, value:any] // ['float[2]', [0, 1]] }
    type pixels = texture |
                  Uint8Array | Float32Array |
                  HTMLImageElement | HTMLCanvasElement | HTMLVideoElement | ImageData | ImageBitmap
    type meshdata = [
        indices   :Uint32Array,  // 3
        positions :Float32Array, // 3
        uvs       :Float32Array, // 2
        normals   :Float32Array, // 3
        tangents  :Float32Array, // 4 (.w is needed for computing bitangent; cross(n, t) * t.w)
        joints    :Uint32Array,  // 4
        weights   :Float32Array  // 4
    ]
```
