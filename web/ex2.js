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
let displace = `
    ${flow}
    float classic_flow(vec2 uv, float time) {
        return flow(uv * 10. + 0.5, 1., 1., radians(45.), time);
    }
    vec4 displace(sampler2D img, vec2 uv, float strength, float time, float scale) {
        float n = classic_flow(uv * scale, time * 1.2);
        return texture(img, uv + strength * 0.005 * sin(n * pi * 2.0));
    }
    vec2 bezier(vec2 a, vec2 b, vec2 c, vec2 d, float t) {
        return pow(1. - t, 3.) * a + 3. * pow(1. - t, 2.) * t * b + 3. * (1. - t) * pow(t, 2.) * c + pow(t, 3.) * d;
    }
    float ease(float x, float y, float a, float b, float t) {
        return bezier(vec2(0.), vec2(x, y), vec2(a, b), vec2(1.), t).y;
    }
`
let proj = `
    mat4 view(float time) {
        float x = 0.;
        float y = 0.;
        float z = 1410.;
        float speed = 0.4;
        float mx = 1.;
        float my = 0.8;
        return inverse(lookat(translate(
            x + cos(time * 0.668 * speed) * 0.104 * z * mx,
            y + sin(time * 0.860 * speed) * 0.080 * z * my,
            z
        ), vec3(0)));
    }
    mat4 proj(float aspect, float time) {
        return perspective(68.621, aspect, 0.1, 100000.) * view(time);
    }
`

let layer = (g, imgs, time, front) => {
    g.draw({
        clear: !front,
        uniforms: {
            time: time * 2,
            image: front? imgs.test : imgs.bg,
            bw: imgs.bw,
            black: imgs.black,
            qdisplace: imgs.displace,
            qmask: imgs.mask
        },
        shader: `
            ${displace}
            ${proj}
            mat4 vertex(vx v) {
                float z = 0.;
                float s = 1920.;
                bool front = ${front};
                if(!front) {
                    z = -8560. + 1860.;
                    s *= 7.1;
                }
                else {
                    z = 140.;
                }
                return proj(v.aspect, v.time) *
                    translate(0., 0., z) * scale(s, s * (1. / v.aspect), 1.);
            }
            float grey(vec3 c) { return (c.r + c.g + c.b)/3.; }
            vec4 pixel(px p) {
                bool front = ${front};
                if(front) {
                    if(texture(p.qmask, p.uv).a < 0.5) return vec4(0);
                }
                vec4 a = texture(p.bw, p.uv);
                vec4 disp = displace(p.image, p.uv, texture(p.qdisplace, p.uv).x, p.time * 0.5, 1.);
                float color = classic_flow(p.uv, p.time) * 0.1 - 0.05;
                color = color + p.uv.y * 0.2 - 0.1;
                color = color + abs(sin(p.time)) * 0.1;
                vec4 b = hue(disp, color);
                float n = classic_flow(p.uv * 2., p.time * -1.);
                if(texture(p.black, p.uv).a > 0.5) {
                    n = ease(1., 0., 1., 0.25, n);
                    if(grey(b.rgb) > 0.2) {
                        b.rgb *= 2.5;
                        a.rgb *= 0.0001;
                    }
                }
                if(!front) a = vec4(0., 0., 0., 1.);
                return blend(b * n, a);
            }
        `
    })
}

export default async ({ path, view }) => {
    let [w, h] = kit.fit(1920, 1080, view.width, view.height)
    view.width = w
    view.height = h
    let g = kit.new(view)
    let gbuf = g.buffer()
    let load = kit.load(g, path)
    let imgs = {
        test: await load.texture('test.png'),
        bg: await load.texture('bg.png'),
        bw: await load.texture('bw.png'),
        black: await load.texture('black.png'),
        displace: await load.texture('displace.png'),
        mask: await load.texture('mask.png')
    }
    return ({ time }) => {
        layer(gbuf, imgs, time, false)
        layer(gbuf, imgs, time, true)
        g.flush(gbuf)
    }
}
