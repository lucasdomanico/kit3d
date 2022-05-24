import kit from '/kit3d.js'

let flow = `
    float flow_vnoise(vec2 p) {
        #define rand(f) fract(sin(f) * 10000.)   
        #define rand2d(x, y) rand(x + y * 100.)    // OG 10000.
        float sw = rand2d(floor(p.x), floor(p.y));
        float se = rand2d(ceil(p.x),  floor(p.y));
        float nw = rand2d(floor(p.x), ceil(p.y));
        float ne = rand2d(ceil(p.x),  ceil(p.y));
        #undef rand
        #undef rand2d
        vec2 inter = smoothstep(0., 1., fract(p));
        float s = mix(sw, se, inter.x);
        float n = mix(nw, ne, inter.x);
        return mix(s, n, inter.y);
    }
    float flow_fbm(vec2 p, float time) {
        float total = 0.0;
        total += flow_vnoise(p       - time);
        total += flow_vnoise(p * 2.  + time) / 2.;
        total += flow_vnoise(p * 4.  - time) / 4.;
        total += flow_vnoise(p * 8.  + time) / 8.;
        total += flow_vnoise(p * 16. - time) / 16.;
        total /= 1. + 1./2. + 1./4. + 1./8. + 1./16.;
        return total;
    }
    float flow(vec2 uv, float aspect, float scale, float angle, float time) {
        angle = -angle + pi * 0.25;  // right
        uv = uv - 0.5;               // pivot center
        if(aspect > 1.) uv.x *= aspect;
        else            uv.y *= 1. / aspect;
        uv = (rotatez(angle) * vec4(uv, 1, 1)).xy;
        uv *= scale;
        float x = flow_fbm(uv, time);
        float y = flow_fbm(uv + 100., time);
        return flow_fbm(uv + vec2(x, y), time);
    }
`
let proj = `
    mat4 view(float time) {
        float x = 0.;
        float y = 0.;
        float z = 850. + 1000.;
        float speed = 0.4;
        float mx = 1.9;
        float my = 1.725;
        return inverse(lookat(translate(
            x + cos(time * 0.668 * speed) * 0.104 * z * mx,
            y + sin(time * 0.860 * speed) * 0.080 * z * my,
            z
        ), vec3(0)));
    }
    mat4 proj(float aspect, float time) {
        return perspective(68.621, aspect, 10., 10000.) * view(time);
    }
`
let plasma = `
    float plasma(vec2 uv, float time, float scale) {
        time *= 2.;
        scale *= 10. + 4.;
        float PI = 3.1415926535897932384626433832795;
        vec2 c = (uv - 0.5) * scale - scale / 2.;
        float v = 0.;
        v += sin(c.x + time);
        v += sin((c.y + time) / 2.);
        v += sin((c.x + c.y + time) / 2.);
        c += scale / 2. * vec2(sin(time / 3.), cos(time / 2.));
        v += sin(sqrt(c.x * c.x + c.y * c.y + 1.) + time);
        v = v / 2.0;
        return v * .25;
    }
`

export default async ({ path, view }) => {
    let [w, h] = kit.fit(1920, 1080, view.width, view.height)
    view.width = w
    view.height = h
    let mode = 0
    // events.click(() => mode = mode? 0 : 1)
    let g = kit.new(view)
    let gbuf = g.buffer()
    let load = kit.load(g, path)
    let circle = await load.texture('circle.png')
    let baseimg = await load.texture('base.png')
    let fontimg = await load.texture('font.png')
    let loveimg = await load.texture('love.png')
    let brushdir = await load.texture('brushdir.png')
    let brush = await load.texture('brush.png')
    let buf = g.buffer()
    return ({ time }) => {
        gbuf.draw({
            clear: true,
            uniforms: { time },
            shader: `
                ${proj}
                mat4 vertex(vx v) {
                    float s = 1920. * 3.1;
                    return proj(v.aspect, v.time) *
                        translate(0., 50., -1000.) *
                        scale(s, s * (1. / v.aspect), 1.);
                }
                ${plasma}
                vec4 pixel(px p) {
                    float c = plasma(p.uv, p.time, 1.);
                    return vec4(hsvtorgb(vec3(0.13 + c * 0.1, 1., 1.)), 1.);
                }            
            `
        })
        gbuf.draw(circles(circle, time, 700, -400, -1, 1.3, mode))
        gbuf.draw(love(baseimg, fontimg, loveimg, time))
        gbuf.draw(circles(circle, time, 100, 400, 1.8,  1.5, mode))
        buf.clear()
        buf.draw(rainbow(brush, brushdir, time, 1.32, 400, -100, 0.6, 0.6))
        buf.draw(rainbow(brush, brushdir, time, 0.42, 600, -200, 0.4, 0.4))
        buf.draw(rainbow(brush, brushdir, time, 2.26, 500, -500, 2.4, 1))
        buf.draw(rainbow(brush, brushdir, time, 0.66, -840, 400, -1.6, -1))
        gbuf.draw({
            uniforms: { image:buf.color(0) },
            shader: `
                vec4 pixel(px p) {
                    return zoomblur(p.image, p.uv, 0.5, 0.5, 0.003);
                }
            `
        })
        g.flush(gbuf)
    }
}

let circles = (sprite, time, instances, z, zsign, scale, mode) => ({
    uniforms: { time, sprite, mode },
    instances,
    shader: `
        ${proj}

        float spanrand_x(int seed, int offset) {
            float f = float(seed) + 1.;
            f *= float(offset) + 1.;
            return fract(sin(f) * 10000.);
        }
        vec2 span_x(int seed, float min, float range, float offset, float time) {
            float life = spanrand_x(seed, 0) * range + min;
            float off = spanrand_x(seed, 1) * offset;
            float n = off + time / life;
            // n = (time + off * life) / life;
            int cycle = int(n);
            float t = fract(n);
            return vec2(cycle, t);
        }

        vec2 particle(int instance, float time) {
            return span_x(instance, 2.5, 3.5, 1., time);
        }
        float rand(int instance, int offset) {
            float f = float(instance) + 1.;
            f *= float(offset) + 1.;
            return fract(sin(f) * 10000.);
        }
        mat4 vertex(vx v) {
            #define randx(offset) rand(v.instance, offset)
            float z = ${z.toFixed(2)};
            float zsign = ${zsign.toFixed(2)};
            float s = ${scale.toFixed(2)};
            float w = 2400. * s;
            float h = 1300. * s;
            vec2 cr = particle(v.instance, v.time);
            int c = int(cr.x);
            float t = cr.y;
            vec4 a = vec4(0., 0., randx(c + 2), randx(c + 3));
            float bz = a.z + randx(c + 7) * zsign;
            vec4 b = vec4((randx(c + 4) - 0.5) * w, (randx(c + 5) - 0.5) * h, bz, randx(c + 7));
            float off = randx(c + 8) * 0.4 + 0.6;
            // vec4 p = mix(a, b, t * (1. - off) + off);
            vec4 p = mix(a, b, rescale(t, 0., 1., off, 1.));
            float n = float(v.instance) / float(v.instances);
            if(n < 0.25) {
                p.w = 0.1;
            }
            p.w *= s;
            return proj(v.aspect, v.time) * translate(p.x, p.y, (p.z - 0.5) * 200. + z) * scale(200., 200., 1.) * scale(p.w, p.w, 1.);
        }
        // pixel shader
        float circle(vec2 uv, float r) {
            float lerp = 0.02;
            vec2 dist = uv - vec2(0.5);
            return 1. - smoothstep(r - r * lerp, r + r * lerp, dot(dist, dist) * 4.0);
        }
        float shape(vec2 uv, float v[10]) {
            float c = 0.;
            for(int i=0; i<10 ;i+=2) {
                c += circle(uv, v[i]);
                c -= circle(uv, v[i + 1]);
            }
            return c;
        }
        float match(vec2 uv, float type, sampler2D sprite) {
            if(type < 0.25) return texture(sprite, uv).a;
            if(type < 0.40) return shape(uv, float[10](1., 0.6,  0.4, 0.3,  0., 0., 0., 0., 0., 0.));
            if(type < 0.60) return shape(uv, float[10](1., 0.97, 0.6, 0.57, 0., 0., 0., 0., 0., 0.));
            if(type < 0.75) return shape(uv, float[10](1., 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55));
            if(type < 0.80) return shape(uv, float[10](1., 0.90, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15));
            if(type < 1.01) return shape(uv, float[10](1., 0.97, 0.8, 0.77, 0.5, 0.35, 0.25, 0.2, 0., 0.));
            return 1.;
        }
        vec4 pixel(px p) {
            float type = float(p.instance) / float(p.instances);
            float c = match(p.uv, type, p.sprite);
            float t = particle(p.instance, p.time).y;
            float a = c * smoothstep(0., 0.3, wave(t)) * 0.4;
            float h = abs(sin(type * 100.14)) * 0.15 + 0.035;
            if(type < 0.25) {
                h = type * 10.02;
                if(type < 0.12) a = a * 1.2;
            }
            return vec4(hsvtorgb(vec3(h, 1., 1.)) * a, a);
        }
    `
})

let love = (base, font, image, time) => ({
    instances: 20,
    uniforms: { time, base, font, image },
    shader: `
        ${proj}
        mat4 vertex(vx v) {
            float z = 400. + (1. - float(v.instance) / float(v.instances)) * -300.;
            float yfix = 20.;
            float size = 1920. * 1.2;
            return proj(v.aspect, v.time) *
                translate(0., yfix, z) *
                scale(size, size * (1. / v.aspect), 1.);
        }
        ${flow}
        float classic_flow(vec2 uv, float time) {
            return flow(uv * 10. + 0.5, 1., 1., radians(45.), time);
        }
        vec4 mist(vec4 c, vec2 uv, float time) {
            float n = classic_flow(uv * 2., time * 1.6 * 0.7 * 2.);
            c = hue(c, n * 0.3 + -0.07);
            return c;
        }
        vec4 pixel(px p) {
            vec4 a = texture(p.base, p.uv);
            a = mist(a, p.uv, p.time);
            a = blend(hue(texture(p.font, p.uv), p.time * 4.), a);
            a = blend(texture(p.image, p.uv), a);
            if(p.instance < p.instances - 1) {
                float t = float(p.instance) / float(p.instances);
                vec3 c = mix(vec3(0.4, 0.3, 0.) * 2., vec3(0.4, 0., 0.), t);
                a = texture(p.image, p.uv);
                a.rgb = c * a.a;
            }
            return a;
        }
    `
})

let rainbow = (brush, brushdir, time, offset, x, y, scalex, scaley) => ({
    instances: 2,
    uniforms: { time, brushdir, brush },
    shader: `
        ${proj}
        mat4 vertex(vx v) {
            vec3 p = v.instance == 0? vec3(0., 0., 0.) : vec3(70., -22., -0.2);
            float x = ${x.toFixed(2)} + p.x;
            float y = ${y.toFixed(2)} + p.y;
            float sx = ${scalex.toFixed(2)} + p.z;
            float sy = ${scaley.toFixed(2)} + p.z;
            return proj(v.aspect, v.time) * translate(x, y, 800.) * scale(711., 481., 1.) * scale(sx, sy, 1.);
        }
        //
        float grey(vec3 c) { return (c.r+c.g+c.b)/3.; }
        float alpha(vec2 uv, float time, sampler2D brushdir) {
            vec4 c = texture(brushdir, uv);
            float w = grey(c.rgb);
            float talpha = 0.5;
            float tvisible = 4.;
            float t = mod(time, talpha + tvisible + talpha);
            if(t < talpha) {
                float v = t/talpha;
                float dif = v - w;
                // if(c.a < 0.99) discard; // bug ...
                if(c.a < 0.99) return 0.;
                if(dif > 0.) return 1.;
                return 0.;
            }
            if(t < talpha + tvisible) return 1.;
            float v = (t - (talpha + tvisible)) / talpha;
            float dif = v - w;
            if(dif < 0.) return 1.;
            return 0.;
        }
        vec4 pixel(px p) {
            float t = p.time + ${offset.toFixed(2)};
            float h = grey(texture(p.brushdir, p.uv).rgb);
            vec4 c = texture(p.brush, p.uv);
            c = hue(c, abs(sin(t)) * -0.84 * h);
            c *= alpha(p.uv, t, p.brushdir) * 0.6;
            return c;
        }
    `
})
