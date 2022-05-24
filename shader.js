let math = `
    const float pi = 3.1415926535897932384626433832795;
    // https://en.wikipedia.org/wiki/Waveform
    // #define sine(x)     smoothstep(0., 1., wave(x))
    // #define square(x)   step(0.5, wave(x))
    // #define sawtooth(x) mod(wave(x), 1.)
    float priv_lwave(float t) {
        return 1. - abs(mod(t, 1.) * 2. - 1.);
    }
    float wave(float t) {
        return smoothstep(0., 1., priv_lwave(t));
        // not the same: ...
        // return sin(t * pi * 2. - pi * 0.5) * 0.5 + 0.5; // CHECK
    }
    /*
        vec2 bezier(vec2 a, vec2 b, vec2 c, vec2 d, float t) {
            return pow(1. - t, 3.) * a + 3. * pow(1. - t, 2.) * t * b + 3. * (1. - t) * pow(t, 2.) * c + pow(t, 3.) * d;
        }
        float ease(float x, float y, float a, float b, float t) {
            return bezier(vec2(0.), vec2(x, y), vec2(a, b), vec2(1.), t).y;
        }
    */
    // rescale: a/b == rescale(a, 0, b, 0, 1)
    float rescale(float x, float amin, float amax, float bmin, float bmax) {
        float a = amax - amin;
        float b = bmax - bmin;
        return (x - amin) * b / a + bmin;
    }
    // float range(float amin, float amax, float bmin, float bmax, float x) {
    //     float a = amax - amin;
    //     float b = bmax - bmin;
    //     return (x - amin) * b / a + bmin;
    // }
`
let matrices = `
    mat4 translate(float x, float y, float z) {
        return mat4(
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            x, y, z, 1
        );
    }
    mat4 rotatex(float a) {
        float c = cos(a);
        float s = sin(a);
        return mat4(
            1, 0, 0, 0,
            0, c, s, 0,
            0,-s, c, 0,
            0, 0, 0, 1
        );
    }
    mat4 rotatey(float a) {
        float c = cos(a);
        float s = sin(a);
        return mat4(
            c, 0,-s, 0,
            0, 1, 0, 0,
            s, 0, c, 0,
            0, 0, 0, 1
        );
    }
    mat4 rotatez(float a) {
        float c = cos(a);
        float s = sin(a);
        return mat4(
            c, s, 0, 0,
           -s, c, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1
        );
    }
    mat4 scale(float x, float y, float z) {
        return mat4(
            x, 0, 0, 0,
            0, y, 0, 0,
            0, 0, z, 0,
            0, 0, 0, 1
        );
    }


    float hfov2v(float hfov, float aspect) {
        return 2. * atan(tan(hfov * 0.5) / aspect);
    }
    float vfov2h(float vfov, float aspect) {
        return 2. * atan(tan(vfov * 0.5) * aspect);
    }
    // float f = 1. / tan(radians(fov) * 0.5);
    // infinite far: [10]: -1, [14]: -2 * near
    mat4 perspective_vfov_r(float vfov, float aspect, float near, float far) {
        float f = tan(pi * 0.5 - 0.5 * vfov);
        float rangeinv = 1.0 / (near - far);
        return mat4(
            f / aspect, 0,                            0,  0,
                     0, f,                            0,  0,
                     0, 0, (near + far) * rangeinv * 1., -1,
                     0, 0,  near * far  * rangeinv * 2.,  0
        );
    }
    mat4 perspective(float fov, float aspect, float near, float far) {
        return perspective_vfov_r(hfov2v(radians(fov), aspect), aspect, near, far);
    }





    // mat4 frustum(float left, float right, float bottom, float top, float near, float far) {
    //     float dx = right - left;
    //     float dy = top - bottom;
    //     float dz = near - far;
    //     return mat4(
    //         2. * near / dx, 0, 0, 0,
    //         0, 2. * near / dy, 0, 0,
    //         (left + right) / dx, (top + bottom) / dy, far / dz, -1,
    //         0, 0, near * far / dz, 0
    //     );
    // }

    mat4 ortho(float left, float right, float top, float bottom, float near, float far) {
        float w = 1. / (right - left);
        float h = 1. / (top - bottom);
        float p = 1. / (far - near);
        float x = (right + left) * w;
        float y = (top + bottom) * h;
        float z = (far + near) * p;
        return mat4(
            2. * w,      0,       0, -x,
                0, 2. * h,       0, -y,
                0,      0, -2. * p, -z,
                0,      0,       0,  1
        );
    }
    const mat4 orthoquad = mat4(
        2., 0,  0,   0,
        0,  2., 0,  -0,
        0,  0, -2., -1.,
        0,  0,  0,   1.
    );
    // const mat4 orthoquad = mat4(
    //     1, 0, 0, -0,
    //     0, 1, 0, -0,
    //     0, 0, -1, -0,
    //     0, 0, 0, 1
    // );




    // vec3 rvec_dir(mat3 m, vec3 dir) {
    //     return vec3(
    //         m[0].x * dir.x + m[1].x * dir.y + m[2].x * dir.z,
    //         m[0].y * dir.x + m[1].y * dir.y + m[2].y * dir.z,
    //         m[0].z * dir.x + m[1].z * dir.y + m[2].z * dir.z
    //     );
    // }
    vec3 position(mat4 m) {
        return m[3].xyz;
    }
    // mat3 rotation(mat3 m, vec3 scale) {
    //     return mat3(m[0] * (1./scale.x), m[1] * (1./scale.y), m[2] * (1./scale.z));
    // }
        //  position
        //  rotation
        //  scaledup
    vec3 get_scale(mat4 m) {
        vec3 s = vec3(length(m[0].xyz), length(m[1].xyz), length(m[2].xyz));
        float det = determinant(m);
        if(det < 0.) s.x = -s.x;
        return s;
    }
    vec3[3] decompose_unsafe(mat4 m) {
        vec3 p = position(m);
        vec3 s = get_scale(m);
        m[0].xyz *= 1. / s.x;
        m[1].xyz *= 1. / s.y;
        m[2].xyz *= 1. / s.z;
		float m11 = m[0].x, m12 = m[1].x, m13 = m[2].x;
        float m21 = m[0].y, m22 = m[1].y, m23 = m[2].y;
		float m31 = m[0].z, m32 = m[1].z, m33 = m[2].z;
        // euler xyz
        float y = asin(clamp(m13, -1., 1.));
        float x, z;
        if(abs(m13) < 0.9999999) {
            x = atan(-m23, m33);
            z = atan(-m12, m11);
        }
        else {
            x = atan(m32, m22);
            z = 0.;
        }
        vec3[3] r;
        r[0] = p;
        r[1] = vec3(x, y, z);
        r[2] = s;
        return r;
    }

    mat4 priv_lookatup(mat4 m, vec3 t, vec3 up) { // missing scale **********************************
        vec3 p = position(m);
        vec3 z = normalize(p - t);
        vec3 x = normalize(cross(up, z));
        vec3 y = normalize(cross(z, x));
        return mat4(
            x[0], x[1], x[2], 0,
            y[0], y[1], y[2], 0,
            z[0], z[1], z[2], 0,
            p[0], p[1], p[2], 1
                // -dot(vec3(x[0], y[0], z[0]), p),
                // -dot(vec3(x[1], y[1], z[1]), p),
                // -dot(vec3(x[2], y[2], z[2]), p),
                // 1
        );
    }
    mat4 lookatup(mat4 m, vec3 t, vec3 up) {
        vec3 s = get_scale(m);
        return priv_lookatup(m, t, up) * scale(s.x, s.y, s.z);
    }
    mat4 lookat(mat4 m, vec3 t) {
        return lookatup(m, t, vec3(0, 1, 0));
    }
    // mat4 lookatz(mat4 m, vec3 t) {
    //     vec3 up = upvec(m);
    //     return lookatup(m, t, up);
    // }

    vec3 upvec(mat4 m) {
        return normalize(vec3(m[1][0], m[1][1], m[1][2]));
    }

    mat4 billboard(mat4 m, mat4 camera) {
        vec3 up = upvec(camera);
        return lookatup(m, -position(camera), up);
    }
    mat4 billboardz(mat4 m, mat4 camera) {
        return lookatup(m, -position(camera), vec3(0, 1, 0));
    }
`
let lights = `
    float diffuse(mat4 light, vec4 pos, vec3 normal) {
        vec3 dist = position(light) - pos.xyz;
        return max(0., dot(normal, normalize(dist)));
    }
    float specular(mat4 light, vec4 pos, vec3 normal, vec3 eyepos, float shininess) {
        vec3 lightpos = position(light);
        vec3 eyedir = normalize(eyepos - pos.xyz);
        vec3 lightdir = normalize(lightpos - pos.xyz);
        vec3 halfv = normalize(eyedir + lightdir);
        return pow(max(dot(normal, halfv), 0.), shininess);
    }
    // https://danielilett.com/2019-06-12-tut2-3-fresnel/
    float fresnel(vec4 pos, vec3 normal, vec3 eyepos) {
        vec3 v = normalize(eyepos - pos.xyz);
        return 1. - dot(normal, v);
    }

    // projective texture mapping
    vec2 projmap(mat4 view, vec4 pos, mat4 proj) {
        vec4 uv = proj * inverse(view) * pos;
        if(uv.w < 0.) return vec2(-1);
        return (uv.xy / uv.w) * 0.5 + 0.5;
    }
    vec2 sphereuv(vec3 n) {
        return vec2(
            0.5 + atan(n.z, n.x) / (pi * 2.),
            0.5 - asin(n.y) / pi
        );
    }
    vec2 spotmap_x(mat4 view, vec4 pos) {
        vec3 dist = normalize(position(view) - pos.xyz);
        vec3 ray = dist * mat3(view);
        return sphereuv(ray * mat3(rotatey(pi / -2.)));
    }
    vec2 spotmap(mat4 view, vec4 pos, float angle) {
        if(angle != 1.) return vec2(-1);
        return spotmap_x(view, pos);
    }
`
let surface = `
    // mr doob
    // vec3 normalgen(sampler2D h, vec2 uv) {
    //     ivec2 size = textureSize(h, 0);
    //     float px = 1. / float(size.x);// + 0.01;
    //     float py = 1. / float(size.y);// + 0.01;
    //     // px = 0.1;
    //     // px = 0.1;
    //  // float h00 = texture(h, uv + vec2(-px, -py)).r;
    //     float h10 = texture(h, uv + vec2( 0., -py)).r;
    //  // float h20 = texture(h, uv + vec2( px, -py)).r;
    //     float h01 = texture(h, uv + vec2(-px,  0.)).r;
    //     float h21 = texture(h, uv + vec2( px,  0.)).r;
    //  // float h02 = texture(h, uv + vec2(-px,  py)).r;
    //     float h12 = texture(h, uv + vec2( 0.,  py)).r;
    //  // float h22 = texture(h, uv + vec2( px,  py)).r;
    //     vec3 c = vec3((h21 - h01) + 0.5, (h12 - h10) + 0.5, 1);
    //     return c;
    // }
    float sampleSobel(sampler2D t, vec2 uv)
    {
        float weight = 1.0;
        float f = 1. - texture(t, uv).r;
        return f * weight - (weight * 0.5);
    }
    // https://www.shadertoy.com/view/Xtd3DS
    vec3 normalgen(sampler2D t, vec2 uv)
    {   
        ivec2 size = textureSize(t, 0);
        float x = 1. / float(size.x);
        float y = 1. / float(size.y);
        // float x = 1. / 900.;
        // float y = 1. / 900.;
        
        // |-1  0  1|
        // |-2  0  2| 
        // |-1  0  1|
        
        float gX = 0.0;
        gX += -1.0 * sampleSobel(t, uv + vec2(-x, -y));
        gX += -2.0 * sampleSobel(t, uv + vec2(-x,  0));
        gX += -1.0 * sampleSobel(t, uv + vec2(-x, +y));
        gX += +1.0 * sampleSobel(t, uv + vec2(+x, -y));
        gX += +2.0 * sampleSobel(t, uv + vec2(+x,  0));
        gX += +1.0 * sampleSobel(t, uv + vec2(+x, +y));
        
        // |-1 -2 -1|
        // | 0  0  0| 
        // | 1  2  1|
        
        float gY = 0.0;
        gY += -1.0 * sampleSobel(t, uv + vec2(-x, -y));
        gY += -2.0 * sampleSobel(t, uv + vec2( 0, -y));
        gY += -1.0 * sampleSobel(t, uv + vec2(+x, -y));
        gY += +1.0 * sampleSobel(t, uv + vec2(-x, +y));
        gY += +2.0 * sampleSobel(t, uv + vec2( 0, +y));
        gY += +1.0 * sampleSobel(t, uv + vec2(+x, +y));
        
    
        vec2 f = vec2(sqrt(gX * gX + gY * gY), atan(-gY, -gX));
        vec2 gradientDirection = f.x * vec2(cos(f.y), sin(f.y));
        vec3 normal = normalize(vec3(gradientDirection, 1.0));
        // normal.x = -normal.x;
        return normal * 0.5 + 0.5;
    }    
    //vec3 normalmap(vec3 normal, vec3 tangent, vec3 bitangent, sampler2D t, vec2 uv, float scale) {
    vec3 normalmap(mat3 tbn, sampler2D t, vec2 uv, float scale) {
        vec3 n = texture(t, uv).xyz * 2.0 - 1.0;
        // n.z *= -0.01;
        n.xy *= scale;
        return normalize(tbn * n);
    }
    vec2 parallax_f(vec2 uv, vec3 viewdir, sampler2D hmap, float scale, float quality) {
        const float min = 10.;
        float max = 512. * quality;
        float n = mix(max, min, abs(dot(vec3(0, 0, 1), viewdir)));  
        float depth = 1. / n;
        float c = 0.;
        vec2 P = viewdir.xy / viewdir.z * scale;
        vec2 delta = P / n;
        vec2 cuv = uv;
        float ch = texture(hmap, cuv).r;
        while(c < ch) {
            cuv -= delta;
            ch = texture(hmap, cuv).r;  
            c += depth;  
        }
        vec2 prev = cuv + delta;
        float after  = ch - c;
        float before = texture(hmap, prev).r - c + depth;
        float weight = after / (after - before);
        vec2 r = prev * weight + cuv * (1. - weight);
        return r;
    }
    // vec2 parallax(vec3 normal, vec3 tangent, vec3 bitangent, vec3 eyepos, vec3 pos, vec2 uv, sampler2D hmap, float scale) {
    // vec2 parallax(vec3 normal, vec3 tangent, vec3 bitangent, sampler2D hmap, vec2 uv, float scale, vec3 pos, vec3 eyepos) {
    vec2 parallax(mat3 tbn, sampler2D t, vec2 uv, float scale, float quality, vec4 pos, vec3 eyepos) {
        mat3 ttbn = transpose(tbn);
        
        ivec2 size = textureSize(t, 0);
        float asp = float(size.x) / float(size.y);

        // asp = 1.;

        eyepos.y *= asp;
        pos.y *= asp;

        vec3 tbnv = ttbn * eyepos;
        vec3 tbnp = ttbn * pos.xyz;
        vec3 vdir = normalize(tbnv - tbnp);
        return parallax_f(uv, vdir, t, scale, quality);
    }
`
let particles = `
    // float[4] catmullrom_init_c(float x0, float x1, float t0, float t1) {
    //     return float[4](
    //         x0,
    //         t0,
    //         -3. * x0 + 3. * x1 - 2. * t0 - t1,
    //         2. * x0 - 2. * x1 + t0 + t1
    //     );
    // }
    // float[4] catmullrom_init(float x0, float x1, float x2, float x3, float tension) {
    //     return catmullrom_init_c(x1, x2, tension * (x2 - x0), tension * (x3 - x1));
    // }
    // float[4] catmullrom_init_nonuni(float x0, float x1, float x2, float x3, float dt0, float dt1, float dt2) {
    //     float t1 = (x1 - x0) / dt0 - (x2 - x0) / (dt0 + dt1) + (x2 - x1) / dt1;
    //     float t2 = (x2 - x1) / dt1 - (x3 - x1) / (dt1 + dt2) + (x3 - x2) / dt2;
    //     t1 *= dt1;
    //     t2 *= dt1;
    //     return catmullrom_init_c(x1, x2, t1, t2);
    // }
    // float catmullrom_calc(float[4] c, float t) {
    //     float t2 = t * t;
    //     float t3 = t2 * t;
    //     return c[0] + c[1] * t + c[2] * t2 + c[3] * t3;
    // }
    // float distance_to_squared(vec3 a, vec3 b) {
    //     float dx = a.x - b.x;
    //     float dy = a.y - b.y;
    //     float dz = a.z - b.z;
    //     return dx * dx + dy * dy + dz * dz;
    // }
    // vec3 catmullrom(vec3[4] p, float tension, float t) {
    //     t = mod(t, 1.);
    //     int LENGTH = 4;
    //     float pp = float(LENGTH - 1) * t;
    //     int ip = int(floor(pp));
    //     float weight = pp - float(ip);
    //     if(weight == 0. && ip == (LENGTH - 1)) {
    //         ip = LENGTH - 2;
    //         weight = 1.;
    //     }
    //     vec3 p0 = (p[0] - p[1]) + p[0];
    //     vec3 p1 = p[ip % LENGTH];
    //     vec3 p2 = p[(ip + 1) % LENGTH];
    //     vec3 p3 = (p[LENGTH - 1] - p[LENGTH - 2]) + p[LENGTH - 1];
    //     // non-uniform
    //     float centripetal = 0.25;
    //     float chordal = 0.5;
    //     float pw = centripetal;
    //     float dt0 = pow(distance_to_squared(p0, p1), pw);
    //     float dt1 = pow(distance_to_squared(p1, p2), pw);
    //     float dt2 = pow(distance_to_squared(p2, p3), pw);
    //     if(dt1 < 1e-4) dt1 = 1.;
    //     if(dt0 < 1e-4) dt0 = dt1;
    //     if(dt2 < 1e-4) dt2 = dt1;
    //     return vec3(
    //         catmullrom_calc(catmullrom_init_nonuni(p0.x, p1.x, p2.x, p3.x, dt0, dt1, dt2), weight),
    //         catmullrom_calc(catmullrom_init_nonuni(p0.y, p1.y, p2.y, p3.y, dt0, dt1, dt2), weight),
    //         catmullrom_calc(catmullrom_init_nonuni(p0.z, p1.z, p2.z, p3.z, dt0, dt1, dt2), weight)
    //     );
    //     // uniform
    //     // tension = 0.7;
    //     return vec3(
    //         catmullrom_calc(catmullrom_init(p0.x, p1.x, p2.x, p3.x, tension), weight),
    //         catmullrom_calc(catmullrom_init(p0.y, p1.y, p2.y, p3.y, tension), weight),
    //         catmullrom_calc(catmullrom_init(p0.z, p1.z, p2.z, p3.z, tension), weight)
    //     );
    // }
    vec3 catmullrom(vec3[4] p, float tension, float t) {
        // Cardinal Spline Matrix
        // https://www.shadertoy.com/view/MlGSz3
        float T = tension;
        mat4 CRM = mat4(-T,        2.0 - T,  T - 2.0,         T,
                               2.0 * T,  T - 3.0,  3.0 - 2.0 * T,  -T,
                              -T,        0.0,      T,               0.0,
                               0.0,      1.0,      0.0,             0.0);
        vec3 G1 = p[0];
        vec3 G2 = p[1];
        vec3 G3 = p[2];
        vec3 G4 = p[3];
        vec3 A = G1 * CRM[0][0] + G2 * CRM[0][1] + G3 * CRM[0][2] + G4 * CRM[0][3];
        vec3 B = G1 * CRM[1][0] + G2 * CRM[1][1] + G3 * CRM[1][2] + G4 * CRM[1][3];
        vec3 C = G1 * CRM[2][0] + G2 * CRM[2][1] + G3 * CRM[2][2] + G4 * CRM[2][3];
        vec3 D = G1 * CRM[3][0] + G2 * CRM[3][1] + G3 * CRM[3][2] + G4 * CRM[3][3];
        return t * (t * (t * A + B) + C) + D;
    }
    // vec3 curvepath(vec3[16] p, int size, float t) {
    //     t = mod(t, 1.);
    //     int i = int(floor(rescale(t, 0., 1., 0., float(size - 3))));
    //     float d = 1. / float(size - 3);
    //     float dt = mod(t, d);
    //     vec3[4] v = vec3[4](p[i], p[i + 1], p[i + 2], p[i + 3]);
    //     float x = 1. / 3.;
    //     return catmullrom(v, 1., rescale(dt, 0., d, x, x + x));
    // }
    vec3 curvepath(vec3[64] p, float[64] ts, int size, float t) {
        t = mod(t, 1.);
        int i = int(floor(rescale(t, 0., 1., 0., float(size - 3))));
        float d = 1. / float(size - 3);
        float dt = mod(t, d);
        vec3[4] v = vec3[4](p[i], p[i + 1], p[i + 2], p[i + 3]);
        return catmullrom(v, ts[i + 1], rescale(dt, 0., d, 0., 1.));
    }
//
//let span =
    float randi2(int a, int b) {
        vec2 st = vec2(a, b);
        return fract(
            sin(
                dot(st, vec2(12.9898, 78.233))
            ) * 43758.5453123
        );
    }
    vec3 span(int seed, float range, float min, float offset, float time) {
        float life = randi2(seed, 0) * range + min;
        float off = randi2(seed, 1) * offset;
        float n = off + time / life;
        int cycle = int(n) + 2; // +2 to avoid overlap of randi2
        float t = fract(n);
        return vec3(cycle, t, seed);
    }
    const int START = 0;
    const int START_RAND = 1;
    const int END = 2;
    const int END_RAND = 3;
    float emit_field(float[4] v, float t, float r1, float r2) {
        return mix(
            v[START] + rescale(r1, 0., 1., -v[START_RAND], v[START_RAND]),
            v[END]   + rescale(r2, 0., 1., -v[END_RAND],   v[END_RAND]),
            t
        );
    }
    mat4 emit(vec3 span, vec3[4] points, float[4] angle, float[4] size) {
        int c = int(span.x);
        float t = span.y;
        int seed = int(span.z);
        #define r() randi2(seed, c++)
        vec3 a = points[START];
        vec3 ar = points[START_RAND];
        vec3 b = points[END];
        vec3 br = points[END_RAND];
        a.x += rescale(r(), 0., 1., -ar.x, ar.x);
        a.y += rescale(r(), 0., 1., -ar.y, ar.y);
        a.z += rescale(r(), 0., 1., -ar.z, ar.z);
        b.x += rescale(r(), 0., 1., -br.x, br.x);
        b.y += rescale(r(), 0., 1., -br.y, br.y);
        b.z += rescale(r(), 0., 1., -br.z, br.z);
        vec3 p_ = mix(a, b, t);
        float an = emit_field(angle, t, r(), r());
        float sz = emit_field(size,  t, r(), r());
        #undef r
        return translate(p_.x, p_.y, p_.z) * rotatez(an) * scale(sz, sz, sz);
    }
    mat4 emitcurve(vec3 span, vec3[64] ps, vec3[64] ds, float[64] ts, int length, float[4] angle, float[4] size) {
        int c = int(span.x);
        float t = span.y;
        int seed = int(span.z);
        #define r() randi2(seed, c++)
        vec3[64] q;
        for(int i = 0; i < length; i++) {
            q[i] = vec3(
                ps[i].x + rescale(r(), 0., 1., -ds[i].x, ds[i].x),
                ps[i].y + rescale(r(), 0., 1., -ds[i].y, ds[i].y),
                ps[i].z + rescale(r(), 0., 1., -ds[i].z, ds[i].z)
            );
        }
        vec3 p_ = curvepath(q, ts, length, t);
        float an = emit_field(angle, t, r(), r());
        float sz = emit_field(size,  t, r(), r());
        #undef r
        return translate(p_.x, p_.y, p_.z) * rotatez(an) * scale(sz, sz, sz);
    }
//
// let slash = 
    float slash_hard(vec4 p_, float a, float b, float c, float d, float time) {
        float t = mod(time, a + b + c + d);
        if(t < a) return (t / a - p_.r) > 0.? p_.a : 0.;
        t -= a;
        if(t < b) return p_.a;
        t -= b;
        if(t < c) return (t / c - p_.r) > 0.? 0. : p_.a;
        return 0.;
    }
    float slash_one(float r, float a, float t, float n) {
        float tn = t / n;
        float f = tn - r;
        float o = 0.5;
        o = mix(o, 0., tn); // float o = 1. - tn;
        if(f < 0.) return 0.;
        if(f < o) {
            return mix(0., a, rescale(f, 0., o, 0., 1.));
        }
        return 1.;
    }
    float slash(float pr, float pa, float a, float b, float c, float d, float time) {
        float t = mod(time, a + b + abs(c) + d);
        if(t < a) {
            return slash_one(pr, pa, t, a);
        }
        t -= a;
        if(t < b) {
            return pa;
        }
        t -= b;
        if(c < 0. && t < -c) {
            return slash_one(pr, pa, -c - t, -c);
        }
        if(t < c) {
            return 1. - slash_one(pr, pa, t, c);
        }
        return 0.;
    }
`
let color = `
    vec3 rgbtohsv(vec3 c) {
        vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
        vec4 p_ = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
        vec4 q = mix(vec4(p_.xyw, c.r), vec4(c.r, p_.yzx), step(p_.x, c.r));
        float d = q.x - min(q.w, q.y);
        float e = 1.0e-10;
        return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
    }
    vec3 hsvtorgb(vec3 c) {
        vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
        vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
        return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
    }
    vec4 hue(vec4 c, float v) {
        vec3 hsv = rgbtohsv(c.rgb);
        hsv.x += v;
        c.rgb = hsvtorgb(hsv);
        return c;
    }
    vec4 invert(vec4 c) {
        return vec4(1.-c.r, 1.-c.g, 1.-c.b, c.a);
    }
`
let filters = `
    // vec4 blur_13(sampler2D image, vec2 uv, vec2 strength) {
    //     vec4 color = vec4(0.0);
    //     vec2 off1 = vec2(1.411764705882353) * strength;
    //     vec2 off2 = vec2(3.2941176470588234) * strength;
    //     vec2 off3 = vec2(5.176470588235294) * strength;
    //     color += texture(image, uv) * 0.1964825501511404;
    //     color += texture(image, uv + off1) * 0.2969069646728344;
    //     color += texture(image, uv - off1) * 0.2969069646728344;
    //     color += texture(image, uv + off2) * 0.09447039785044732;
    //     color += texture(image, uv - off2) * 0.09447039785044732;
    //     color += texture(image, uv + off3) * 0.010381362401148057;
    //     color += texture(image, uv - off3) * 0.010381362401148057;
    //     return color;
    // }
    // vec4 hblur(sampler2D t, vec2 uv, float strength) {
    //     return blur_13(t, uv, vec2(strength, 0));
    // }
    // vec4 vblur(sampler2D t, vec2 uv, float strength) {
    //     return blur_13(t, uv, vec2(0, strength));
    // }

    vec4 blur_13(sampler2D image, vec2 uv, vec2 direction) {	
        vec4 color = vec4(0.0);	
        vec2 off1 = vec2(1.411764705882353) * direction;	
        vec2 off2 = vec2(3.2941176470588234) * direction;	
        vec2 off3 = vec2(5.176470588235294) * direction;	
        color += texture(image, uv) * 0.1964825501511404;	
        color += texture(image, uv + off1) * 0.2969069646728344;	
        color += texture(image, uv - off1) * 0.2969069646728344;	
        color += texture(image, uv + off2) * 0.09447039785044732;	
        color += texture(image, uv - off2) * 0.09447039785044732;	
        color += texture(image, uv + off3) * 0.010381362401148057;	
        color += texture(image, uv - off3) * 0.010381362401148057;	
        return color;	
    }	
    vec4 hblur(sampler2D t, vec2 uv, float strength) {	
        return blur_13(t, uv, vec2(strength, 0));	
    }	
    vec4 vblur(sampler2D t, vec2 uv, float strength) {
        return blur_13(t, uv, vec2(0, strength));
    }
//
//let zoomblur = 
    vec4 zoomblur(sampler2D img, vec2 uv, float x, float y, float strength) {
        vec2 inc = (vec2(x, 1. - y) - uv) * strength;
        vec4 sum;
        sum += texture(img, uv - inc * 4.) * 0.051;
        sum += texture(img, uv - inc * 3.) * 0.0918;
        sum += texture(img, uv - inc * 2.) * 0.12245;
        sum += texture(img, uv - inc * 1.) * 0.1531;
        sum += texture(img, uv + inc * 0.) * 0.1633;
        sum += texture(img, uv + inc * 1.) * 0.1531;
        sum += texture(img, uv + inc * 2.) * 0.12245;
        sum += texture(img, uv + inc * 3.) * 0.0918;
        sum += texture(img, uv + inc * 4.) * 0.051;
        return sum;
    }
    // vec4 zoomblur2(sampler2D img, vec2 uv, float x, float y, float sa, float sb) {
    //     vec2 inc = (vec2(x, 1. - y) - uv) * vec2(sa, sb);
    //     vec4 sum;
    //     sum += texture(img, uv - inc * 4.) * 0.051;
    //     sum += texture(img, uv - inc * 3.) * 0.0918;
    //     sum += texture(img, uv - inc * 2.) * 0.12245;
    //     sum += texture(img, uv - inc * 1.) * 0.1531;
    //     sum += texture(img, uv + inc * 0.) * 0.1633;
    //     sum += texture(img, uv + inc * 1.) * 0.1531;
    //     sum += texture(img, uv + inc * 2.) * 0.12245;
    //     sum += texture(img, uv + inc * 3.) * 0.0918;
    //     sum += texture(img, uv + inc * 4.) * 0.051;
    //     return sum;
    // }
    float sobel(sampler2D t, vec2 uv, float x, float y) { // x and y are strength
        vec4 h = vec4(0);
        h -= texture(t, vec2(uv.x - x, uv.y - y)) * 1.;
        h -= texture(t, vec2(uv.x - x, uv.y    )) * 2.;
        h -= texture(t, vec2(uv.x - x, uv.y + y)) * 1.;
        h += texture(t, vec2(uv.x + x, uv.y - y)) * 1.;
        h += texture(t, vec2(uv.x + x, uv.y    )) * 2.;
        h += texture(t, vec2(uv.x + x, uv.y + y)) * 1.;
        vec4 v_ = vec4(0);
        v_ -= texture(t, vec2(uv.x - x, uv.y - y)) * 1.;
        v_ -= texture(t, vec2(uv.x    , uv.y - y)) * 2.;
        v_ -= texture(t, vec2(uv.x + x, uv.y - y)) * 1.;
        v_ += texture(t, vec2(uv.x - x, uv.y + y)) * 1.;
        v_ += texture(t, vec2(uv.x    , uv.y + y)) * 2.;
        v_ += texture(t, vec2(uv.x + x, uv.y + y)) * 1.;
        vec3 edge = sqrt(h.rgb * h.rgb + v_.rgb * v_.rgb);
        return (edge.x + edge.y + edge.z) / 3.;
    }
    // https://www.shadertoy.com/view/XlsXRB
    float outline_x(sampler2D t, vec2 uv, float samples_dx, bool inside) {
        vec2 size = vec2(textureSize(t, 0));
        ivec2 iuv = ivec2(int(uv.x * size.x), int(uv.y * size.y));
        float samples = round(samples_dx * size.x);
        int isamples = int(samples);
        float d = samples;
        for(int x = -isamples; x != isamples; x++) {
            for(int y = -isamples; y != isamples; y++) {
                float a = texelFetch(t, iuv + ivec2(x, y), 0).a;
                if(inside) a = 1. - a;
                // if(dot(normalize(vec2(x, y)), normalize(vec2(0, -5))) < 0.75) continue;
                if(a > 0.5) {
                    d = min(d, length(vec2(x, y)));
                }
            }
        }
        d = clamp(d, 0., samples) / samples;
        // d = clamp(d, 0., 1.); // may be over -0.00000001
        return 1. - d;
    }
    float outline(sampler2D t, vec2 uv, float samples_dx) {
        return outline_x(t, uv, samples_dx, false);
    }
`
let misc = `
    float aspect(sampler2D t) {
        ivec2 size = textureSize(t, 0);
        return float(size.x) / float(size.y);
    }
    bool wrap(vec2 uv) {
        return uv.x < 0. || uv.x > 1. || uv.y < 0. || uv.y > 1.;
    }

    // mat4 quat2mat(vec4 q) {
    //     float x2 = q.x + q.x, y2 = q.y + q.y, z2 = q.z + q.z;
    //     float xx = q.x * x2,  xy = q.x * y2,  xz = q.x * z2;
    //     float yy = q.y * y2,  yz = q.y * z2,  zz = q.z * z2;
    //     float wx = q.w * x2,  wy = q.w * y2,  wz = q.w * z2;
    //     vec3 s = vec3(1);
    //     mat4 m = mat4(1);
    //     m[0][0] = (1. - (yy + zz)) * s.x;
    //     m[1][0] = (xy + wz) * s.x;
    //     m[2][0] = (xz - wy) * s.x;
    //     m[0][1] = (xy - wz) * s.y;
    //     m[1][1] = (1. - (xx + zz)) * s.y;
    //     m[2][1] = (yz + wx) * s.y;
    //     m[0][2] = (xz + wy) * s.z;
    //     m[1][2] = (yz - wx) * s.z;
    //     m[2][2] = (1. - (xx + yy)) * s.z;
    //     return transpose(m); /////////////////////////////////////////////////////////
    // }
    // mat4 clip_x(sampler2D s, uvec4 joints, vec4 weights, float time) {
    //     vec4 head = texelFetch(s, ivec2(0, 0), 0);
    //     float length = head.x;
    //     // if(length == 0.) return mat4(1);
    //     float fps = head.y;
    //     int bones = int(head.z);
    //     if(bones <= 1) return mat4(1);
    //     float t = mod(time, length);
    //     int frame = int(t * fps);
    //     mat4 r = mat4(0);
    //     int size = textureSize(s, 0).x;
    //     for(int i = 0; i < 4; i++) {
    //         int p = 1 + frame * bones * 2 + (int(joints[i]) + 1) * 2;

    //         int x = p % size;
    //         int y = p / size;
    //         int x2 = (p + 1) % size;
    //         int y2 = (p + 1) / size;


    //         // int x = int(floor(mod(float(p), float(size))));
    //         // int y = int(floor(   float(p) / float(size)));
    //         // int x2 = int(floor(mod(float(p + 1), float(size))));
    //         // int y2 = int(floor(   float(p + 1) / float(size)));


    //         // x = p - (y * size);
    //         vec3 a =  texelFetch(s, ivec2(x, y), 0).xyz;
    //         vec4 aq = texelFetch(s, ivec2(x2, y2), 0);

    //         // vec3 a =  texelFetch(s, ivec2(p, 0), 0).xyz;
    //         // vec4 aq = texelFetch(s, ivec2(p + 1, 0), 0);
    //         r += (translate(a.x, a.y, a.z) * quat2mat(aq)) * weights[i];
    //     }
    //     return r;
    // }

    float texel(sampler2D s, int p) {
        int channels = 4;
        int width = textureSize(s, 0).x * channels;
        int x = p % width;
        int y = p / width;
        vec4 tex = texelFetch(s, ivec2(x / channels, y), 0);
        int offset = x % channels;
        return tex[offset];
    }
    mat4 clip_item(sampler2D s, int p, int size) {
        int x1 = p % size;
        int y1 = p / size;
        int x2 = (p + 1) % size;
        int y2 = (p + 1) / size;
        int x3 = (p + 2) % size;
        int y3 = (p + 2) / size;
        int x4 = (p + 3) % size;
        int y4 = (p + 3) / size;
        vec4 a = texelFetch(s, ivec2(x1, y1), 0);
        vec4 b = texelFetch(s, ivec2(x2, y2), 0);
        vec4 c = texelFetch(s, ivec2(x3, y3), 0);
        vec4 d = texelFetch(s, ivec2(x4, y4), 0);
        return mat4(a, b, c, d);
    }
    mat4 clip(sampler2D s, uvec4 joints, vec4 weights, float time) {
        vec4 head = texelFetch(s, ivec2(0, 0), 0);
        float length = head.x;
        float fps = head.y;
        int stride = int(head.z);
        float t = mod(time, length);
        int frame = int(t * fps);
        int size = textureSize(s, 0).x;
        int p = 1 + frame * stride * 4 + 0;
        mat4 r = clip_item(s, p, size);
        if(stride <= 1) return r;
        mat4 q = mat4(0);
        for(int i = 0; i < 4; i++) {
            int p = 1 + frame * stride * 4 + (int(joints[i]) + 1) * 4;
            q += clip_item(s, p, size) * weights[i];
        }
        return r * q;
    }

// dist
    // https://www.iquilezles.org/www/articles/intersectors/intersectors.htm
    // https://www.geometrictools.com/Source/Distance3D.html
    // r = 0.0025
    // distrayseg
    // if m is 1, returns depth
    //    when 0, just 0 (overlap)
    float distrayseg(vec3 ro, vec3 rd, vec3 pa, vec3 pb, float r, float m, float none) {
        vec3 ba = pb - pa;
        vec3 oa = ro - pa;
        float baba = dot(ba, ba);
        float bard = dot(ba, rd);
        float baoa = dot(ba, oa);
        float rdoa = dot(rd, oa);
        float oaoa = dot(oa, oa);
        float a = baba        - bard * bard;
        float b = baba * rdoa - baoa * bard;
        float c = baba * oaoa - baoa * baoa - r * r * baba;
        float h = b * b - a * c;
        if(h >= 0.) {
            float t = (-b - sqrt(h)) / a;
            float y = baoa + t * bard;
            if(y > 0. && y < baba) { // body
                return t * m;
            }
            vec3 oc = (y <= 0.)? oa : ro - pb; // caps
            b = dot(rd, oc);
            c = dot(oc, oc) - r * r;
            h = b * b - c;
            if(h > 0.) return -b - sqrt(h);
        }
        return none;
    }

    ///////////////// sprite /////////////////////
    mat4 sprite(float x, float y, float w, float h, float r, float aspect) {
        mat4 m = mat4(1);
        m = mat4(1)
            * scale(1., aspect, 1.)
            * rotatez(r)
            * scale(1., 1. / (w / h), 1.)
            * scale(w, w, w)
            ;
        return orthoquad * m;
    }
    // docs:
    // float w = 0.25;
    // float h = v.mode == 0.? w : w * (1. / aspect(v.img));
    // return sprite(0.5, 0.5, w, h, v.time * 0.4, v.aspect)
    /*
        1 / (w / h)
        1 / (w / (w * (1. / haspect)))
        1 / (w / (w * (1. / (w / h))))
    */
    mat4 spriteh_(float x, float y, float w, float haspect, float r, float aspect) {
        // return sprite(x, y, w, w * (1. / haspect), r, aspect);
        return sprite(x, y, w, w / haspect, r, aspect);
    }


`
let blend = (name, f) => `
    float ` + name + `_f(float Sc, float Dc) {
        return ` + f + `;
    }
    vec4 ` + name + `(vec4 Sca, vec4 Dca) {
        vec3 Sc = Sca.a == 0.? vec3(0) : Sca.rgb / Sca.a;
        vec3 Dc = Dca.a == 0.? vec3(0) : Dca.rgb / Dca.a;
        float Sa = Sca.a;
        float Da = Dca.a;
        float X = 1.;
        float Y = 1.;
        float Z = 1.;
        float r = ` + name + `_f(Sc.r, Dc.r) * Sa * Da + Y * Sca.r * (1.-Da) + Z * Dca.r * (1.-Sa);
        float g = ` + name + `_f(Sc.g, Dc.g) * Sa * Da + Y * Sca.g * (1.-Da) + Z * Dca.g * (1.-Sa);
        float b = ` + name + `_f(Sc.b, Dc.b) * Sa * Da + Y * Sca.b * (1.-Da) + Z * Dca.b * (1.-Sa);
        float a = X * Sa * Da  + Y * Sa  * (1.-Da) + Z * Da  * (1.-Sa);
        return vec4(r, g, b, a);
    }
`
let blendmodes = `
    ` + blend('blend', `Sc`) + `
    // dissolve
    ` + blend('darken',  `min(Sc, Dc)`) + `
    ` + blend('multiply', `Sc * Dc`) + `
    ` + blend('colorburn', `(Sc == 0.)? 0. : 1. - min(1., (1. - Dc) / Sc)`) + `
    ` + blend('linearburn', `max(Dc + Sc - 1., 0.)`) + `
    // darker color
    ` + blend('lighten', `max(Sc, Dc)`) + `
    ` + blend('screen', `Sc + Dc - (Sc * Dc)`) + `
    ` + blend('colordodge', `(Sc == 1.)? 1. : min(1., Dc / (1. - Sc))`) + `
    ` + blend('addition', `Sc + Dc`) + ` // linear dodge
    // lighter color
    ` + blend('overlay', `
        (2. * Dc <= 1.)?
            2. * Sc * Dc
        :
            1. - 2. * (1. - Dc) * (1. - Sc)
    `) + `
    ` + blend('softlight', `
        (2. * Sc <= 1.)?
            Dc - (1. - 2. * Sc) * Dc * (1. - Dc)
        : (2. * Sc > 1. && 4. * Dc <= 1.)?
            Dc + (2. * Sc - 1.) * (4. * Dc * (4. * Dc + 1.) * (Dc - 1.) + 7. * Dc)
        :
            Dc + (2. * Sc - 1.) * (pow(Dc, 0.5) - Dc)`) + `
    ` + blend('hardlight', `(2. * Sc <= 1.)? 2. * Sc * Dc : 1. - 2. * (1. - Dc) * (1. - Sc)`) + `
    // vividlight
    // linearlight
    // pinlight
    // hardmix
    ` + blend('difference', `abs(Dc - Sc)`) + `
    ` + blend('exclusion',  `Sc + Dc - 2. * Sc * Dc`) + `
    ` + blend('subtract',   `Dc - Sc`) + `
    // divide
    // hue
    // saturation
    // color
    // luminosity
`
let ETC = `
    vec3[2] ray(mat4 proj, mat4 camera, vec2 uv) {
        proj[2][2] = -1.;
        proj[3][2] = -1.;
        vec3 ro = position(camera);
        vec4 ndc = vec4(uv * 2. - 1., 1., 1.);
        vec3 rd = normalize((camera * inverse(proj) * ndc).xyz);
        return vec3[2](ro, rd);
    }

    // i is image, s screen
    vec2 uvfit(vec2 uv, float i, float s) {
        uv -= 0.5;
        if(s > i) uv.x = s * uv.x / i;
        else      uv.y = (1. / s) * uv.y / (1. / i);
        uv += 0.5;
        return uv;
    }
    vec2 uvscale(vec2 uv, float n) {
        uv -= 0.5;
        uv *= 1. / n;
        uv += 0.5;
        return uv;
    }

    float disp(sampler2D disp, vec2 uv, float time, float scale) {
        uv *= scale;
        uv += time;
        uv.x = mod(uv.x, 1.);
        uv.y = mod(uv.y, 1.);
        return textureLod(disp, uv, 0.).x;                   
    }
`
let shaderlib = `
    ` + math + `
    ` + matrices + `
    ` + lights + `
    ` + surface + `
    ` + particles + `
    ` + color + `
    ` + filters + `
    ` + misc + `
    ` + blendmodes + `
    ` + ETC + `
`

let shadertag = '//// code'

let shader = (code, types, global) => {
    let $code = code + (code.includes(' vertex(')? '' : //# includes
    `
        mat4 vertex(vx a) {
            return orthoquad;
        }    
    `)
    let vmats = $code.includes('mat4 vertex(')?    1 : //# includes
                $code.includes('mat4[3] vertex(')? 2 : //# includes
                0
    let pvm = vmats == 2
    let pcols = $code.includes('vec4 pixel(')?     1 : //# includes
                // $code.includes('vec4[2] pixel(')?  2 :
                // $code.includes('vec4[3] pixel(')?  3 :
                // $code.includes('vec4[4] pixel(')?  4 :
                $code.includes('vec5 pixel(')?    5 : //# includes
                // $code.includes('vec4f[2] pixel(')? 6 :
                // $code.includes('vec4f[3] pixel(')? 7 :
                // $code.includes('vec4f[4] pixel(')? 8 :
                0
    let uniforms = types.map(([k, t]) => 'uniform ' + t + ' ' + global + k + ';').join('\n')         //# map join
    let args =     types.map(([k, t]) => t === 'sampler2D'? '' : t + ' ' + k + ';').join('\n')       //# map join
    let init =     types.map(([k, t], i) => t === 'sampler2D'? '' : ', ' + global + k + '').join('') //# map join
    let ret =      types.map(([k, t]) => t === 'sampler2D'? k : '').filter(_ => _).join('|') //# map filter join
    let re = new RegExp('(^|\\W)(v|p|a)\\.(' + ret + ')(\\W|$)', 'g') //# RegExp
    $code = $code.replace(re, (m, _, a, b, c) => _ + global + b + c)  //# replace
    // console.log('CODE ', $code)
    let glsl = `#version 300 es
        precision highp float;
        precision highp int;
        ` + shaderlib + `
        ` + shadertag + `
        ` + uniforms + `
        struct vx {
            int instance;
            uvec4 joints;
            vec4 weights;
            ` + args + `
        };
        struct px {
            int instance;
            vec2 uv;
            bool frontfacing;
            float depth;
            ` + (pvm? `
                vec4 pos;
                vec3 normal;
                mat3 tbn;
                vec3 eyepos;`
            : '') + `
            ` + args + `
        };
        struct vec5 { vec4 _v4; float _f; };
        // f -> nowrite
    `
    let vertex = glsl + `
        #define discard 0.
        ` + $code + `
        layout(location=0) in  vec4 a_pos;
        layout(location=1) in  vec2 a_uv;
        layout(location=2) in  vec3 a_normal;
        layout(location=3) in  vec4 a_tangent;
        layout(location=4) in uvec4 a_joints;
        layout(location=5) in  vec4 a_weights;
        out vec4 v_pos;
        out vec2 v_uv;
        out vec3 v_normal;
        out vec3 v_tangent;
        out vec3 v_bitangent;
        flat out vec3 v_eyepos;
        flat out int  v_instance;
        void main()	{
            vx args = vx(
                gl_InstanceID,
                a_joints,
                a_weights
                ` + init + `
            );            
            mat4 m = mat4(1);
            mat4 v = mat4(1);
            mat4 p = mat4(1);
            ` + (pvm? `
                mat4[3] pvm = vertex(args);
                p = pvm[0];
                v = pvm[1];
                m = pvm[2];`
            : `
                m = vertex(args);
            `) + `
            gl_Position = (p * v * m) * a_pos;
            v_uv = a_uv;
            v_instance = gl_InstanceID;
            ` + (pvm? `
                v_pos = m * a_pos;
                mat3 nm = mat3(transpose(inverse(m)));
                v_normal = normalize(nm * a_normal);
                v_tangent = normalize(nm * a_tangent.xyz);
                v_bitangent = normalize(cross(v_normal, v_tangent) * a_tangent.w);
                v_eyepos = position(inverse(v));`
            : '') + `
        }`
    let pixel = glsl + `
        ` + $code + `
        in vec4 v_pos;
        in vec2 v_uv;
        in vec3 v_normal;
        in vec3 v_tangent;
        in vec3 v_bitangent;
        flat in vec3 v_eyepos;
        flat in int v_instance;
        out vec4[1] color; // resizes?
        void main() {
            vec3 normal = normalize(v_normal);
            vec3 tangent = normalize(v_tangent);
            vec3 bitangent = normalize(v_bitangent);
            px args = px(
                v_instance,
                v_uv,
                gl_FrontFacing,
                gl_FragCoord.z
                // 1.0 - (gl_FragCoord.z / gl_FragCoord.w) / 10000.
                ` + (pvm? `,
                    v_pos,
                    normal,
                    mat3(tangent, bitangent, normal),
                    v_eyepos`
                : '') + `
                ` + init + `
            );
            ` +  (pcols == 1? `
                vec4 c = pixel(args);
                color[0] = c;
            ` :
                pcols == 2? `
                vec4[2] c = pixel(args);
                color[0] = c[0];
                color[1] = c[1];
            ` :
                pcols == 5? `
                vec5 c = pixel(args);
                color[0] = c._v4;
                gl_FragDepth = c._f;
            ` :
               (() => { throw pcols })()
            ) + `
        }`
    // console.log(vertex, pixel)
    return [vertex, pixel]
}

export default shader