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
`
let proj = `
    mat4 view(float time) {
        float x = -10.;
        float y = -25.;
        float z = 800.;
        float speed = 0.4 * 3.;
        float mx = 1. * 0.4 * 2.;
        float my = 1.25 * 0.2 * 2.;
        return inverse(lookat(translate(
            x + cos(time * 0.668 * speed) * 0.104 * z * mx,
            y + sin(time * 0.860 * speed) * 0.080 * z * my,
            z
        ), vec3(0., 0., 200.)));
    }
    mat4 proj(float aspect, float time) {
        return perspective(68.621, aspect, 0.1, 10000.) * view(time);
    }
`

export default async ({ path, view }) => {
    let [w, h] = kit.fit(1920, 1080, view.width, view.height)
    view.width = w
    view.height = h
    let g = kit.new(view)
    let load = kit.load(g, path)
    let bg = await load.texture('bg.png')
    let paul = await load.texture('paul.png')
    let paulhair = await load.texture('paulhair.png')
    let paulface = await load.texture('paulface.png')
    let linda = await load.texture('linda.png')
    let lindahair = await load.texture('lindahair.png')
    let lindaface = await load.texture('lindaface.png')
    let buf = g.buffer()
    let gbuf = g.buffer()
    let prevs = [g.buffer(), g.buffer()]
    return ({ time }) => {
        buf.draw({
            clear: true,
            uniforms: { time, bg },
            shader: `
                ${proj}
                mat4 vertex(vx v) {
                    float size = 4992.;
                    return proj(v.aspect, v.time) *
                        translate(0., 0., -1600.) *
                        scale(size, size * (1. / v.aspect), 1.);
                }
                vec4 pixel(px p) {
                    float t = sin(p.time) / 2. + 0.5;
                    vec4 bgc = vec4(0., 0., 0., 1.);
                    return hue(blend(zoomblur(p.bg, p.uv, t, t, t * 0.004), bgc), 0.3);
                }
            `,
        })
        buf.draw({
            uniforms: { time, image:paul, displacep:paulhair, face:paulface },
            shader: `
                ${proj}
                mat4 vertex(vx v) {
                    float size = 1152.;
                    return proj(v.aspect, v.time) *
                        scale(size, size * (1. / v.aspect), 1.);
                }
                ${MscBRs}
                ${displace}
                vec4 pixel(px p) {
                    vec4 a = invert(MscBRs(p.uv * vec2(0.5, 1.), p.aspect, p.time * 1.5));
                    vec4 b = displace(p.image, p.uv, texture(p.displacep, p.uv).x * 2., p.time + 1., 1.);
                    if(b.a < 0.1) return vec4(0);
                    vec4 f = texture(p.face, p.uv).a > 0.1? vec4(1., 0., 0., 0.7) : vec4(0);
                    a = blend(f, a);
                    return difference(b, a);
                }
            `
        })
        buf.draw({
            uniforms: { time, image:linda, displacep:lindahair, face:lindaface },
            shader: `
                ${proj}
                mat4 vertex(vx v) {
                    float size = 1728.;
                    return proj(v.aspect, v.time) *
                        translate(50., 0., -200.) *
                        scale(size, size * (1. / v.aspect), 1.);
                }
                ${MscBRs}
                ${displace}
                vec4 pixel(px p) {
                    vec4 a = invert(MscBRs(p.uv * vec2(0.5, 1.), p.aspect, p.time * 0.5));
                    vec4 b = displace(p.image, p.uv, texture(p.displacep, p.uv).x * 2., p.time + 1., 1.);
                    if(b.a < 0.1) return vec4(0);
                    vec4 f = texture(p.face, p.uv).a > 0.1? vec4(1., 0., 0., 0.7) : vec4(0);
                    a = blend(f, a);
                    float huet = abs(sin(p.time)) * 0.1;
                    return hue(difference(b, a), 0.60 + huet);
                }
            `
        })
        prevs[0].draw({
            uniforms: { image:buf.color(0), prev:prevs[1].color(0) },
            shader: `
                vec4 pixel(px p) {
                    vec4 a = texture(p.image, p.uv);
                    vec4 b = texture(p.prev, p.uv) * 0.97;
                    if(b.a < 0.03) b = vec4(0.);
                    return blend(b, a);
                }
            `,     
        })
        prevs = [prevs[1], prevs[0]]
        gbuf.draw({
            clear: true,
            uniforms: { t:prevs[1].color(0) },
            shader: `
                vec4 pixel(px p) {
                    return texture(p.t, p.uv);
                }
            `
        })
        g.flush(gbuf)
    }
}

let MscBRs = `
    // @lsdlive
    // This was my shader for the shader showdown at Outline demoparty 2018 in Nederland.
    // Shader showdown is a live-coding competition where two participants are
    // facing each other during 25 minutes.
    // (Round 1)
    // I don't have access to the code I typed at the events, so it might be
    // slightly different.
    // Original algorithm on shadertoy from fb39ca4: https://www.shadertoy.com/view/4dX3zl
    // I used the implementation from shane: https://www.shadertoy.com/view/MdVSDh
    // Thanks to shadertoy community & shader showdown paris.
    // This is under CC-BY-NC-SA (shadertoy default licence)
    mat2 r2d(float a) {
        float c = cos(a), s = sin(a);
        return mat2(c, s, -s, c);
    }
    vec2 path(float t) {
        float a = sin(t*.2 + 1.5), b = sin(t*.2);
        return vec2(2.*a, a*b);
    }
    float g = 0.;
    float de(vec3 p, float time) {
        p.xy -= path(p.z);
        float d = -length(p.xy) + 4.; // tunnel (inverted cylinder)
        p.xy += vec2(cos(p.z + time)*sin(time), cos(p.z + time));
        p.z -= 6. + time * 6.;
        d = min(d, dot(p, normalize(sign(p))) - 1.); // octahedron (LJ's formula)
        // I added this in the last 1-2 minutes, but I'm not sure if I like it actually!
        // Trick inspired by balkhan's shadertoys.
        // Usually, in raymarch shaders it gives a glow effect,
        // here, it gives a colors patchwork & transparent voxels effects.
        g += .015 / (.01 + d * d);
        return d;
    }
    vec4 MscBRs(vec2 uv, float aspect, float time) {   
        uv = vec2(uv.x - 0.5, uv.y - 0.5);
        uv.x *= aspect;
        float dt = time * 6.;
        vec3 ro = vec3(0, 0, -5. + dt);
        vec3 ta = vec3(0, 0, dt);
        ro.xy += path(ro.z);
        ta.xy += path(ta.z);
        vec3 fwd = normalize(ta - ro);
        vec3 right = cross(fwd, vec3(0, 1, 0));
        vec3 up = cross(right, fwd);
        vec3 rd = normalize(fwd + uv.x*right + uv.y*up);
        rd.xy *= r2d(sin(-ro.x / 3.14)*.3);
        // Raycast in 3d to get voxels.
        // Algorithm fully explained here in 2D (just look at dde algo):
        // http://lodev.org/cgtutor/raycasting.html
        // Basically, tracing a ray in a 3d grid space, and looking for 
        // each voxel (think pixel with a third dimension) traversed by the ray.
        vec3 p = floor(ro) + .5;
        vec3 mask;
        vec3 drd = 1. / abs(rd);
        rd = sign(rd);
        vec3 side = drd * (rd * (p - ro) + .5);
        float t = 0., ri = 0.;
        for(float i = 0.; i < 1.; i += .01) {
            ri = i;
            // sphere tracing algorithm (for comparison)
            // p = ro + rd * t;
            // float d = de(p, time);
            // if(d<.001) break;
            // t += d;
            if (de(p, time) < 0.) break; // distance field
            // we test if we are inside the surface
            mask = step(side, side.yzx) * step(side, side.zxy);
            // minimum value between x,y,z, viewput 0 or 1
            side += drd * mask;
            p += rd * mask;
        }
        t = length(p - ro);
        vec3 c = vec3(1) * length(mask * vec3(1., .5, .75));
        c = mix(vec3(.2, .2, .7), vec3(.2, .1, .2), c);
        c += g * .4;
        c.r += sin(time)*.2 + sin(p.z*.5 - time * 6.); // red rings
        c = mix(c, vec3(.2, .1, .2), 1. - exp(-.001*t*t)); // fog
        return vec4(c, 1.0);
    }
`
