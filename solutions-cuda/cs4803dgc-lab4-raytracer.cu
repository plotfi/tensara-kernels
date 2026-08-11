// CS4803 (Spring 2010) Lab 4 — CUDA triangle ray tracer.
// Ported from CS4803VGCD/lab4/raytracer.cu. The only change vs the original is
// mechanical: the deprecated `texture<float4,1>` + tex1Dfetch (removed in CUDA
// 12) are replaced by a plain `const float4* tris` global pointer. All the ray
// math (box/triangle/sphere intersection, diffuse+specular shading, shadow rays,
// stochastic anti-aliasing) is unchanged. Triangles are stored as 3 float4 each:
// (v0, edge1=v1-v0 [w>0 => reflective], edge2=v2-v0).
#include <cuda_runtime.h>
#include <limits.h>
#include "../kernel-implementation/cutil_math_min.cuh"

__device__ int rgbToInt(float r, float g, float b) {
    r = clamp(r, 0.0f, 255.0f); g = clamp(g, 0.0f, 255.0f); b = clamp(b, 0.0f, 255.0f);
    return (int(r) << 16) | (int(g) << 8) | int(b);
}

struct Ray {
    __device__ Ray() {}
    __device__ Ray(const float3 &o, const float3 &d) {
        ori = o; dir = normalize(d);
        inv_dir = make_float3(1.0f/dir.x, 1.0f/dir.y, 1.0f/dir.z);
    }
    float3 ori, dir, inv_dir;
};

struct HitRecord {
    __device__ HitRecord() { t = UINT_MAX; hit_index = -1; color = make_float3(0,0,0); }
    __device__ void resetT() { t = UINT_MAX; hit_index = -1; }
    float t; float3 color; float3 normal; int hit_index;
};

__device__ int RayBoxIntersection(const float3 &BBMin, const float3 &BBMax, const float3 &RayOrg,
                                  const float3 &RayDirInv, float &tmin, float &tmax) {
    float l1 = (BBMin.x - RayOrg.x) * RayDirInv.x, l2 = (BBMax.x - RayOrg.x) * RayDirInv.x;
    tmin = fminf(l1, l2); tmax = fmaxf(l1, l2);
    l1 = (BBMin.y - RayOrg.y) * RayDirInv.y; l2 = (BBMax.y - RayOrg.y) * RayDirInv.y;
    tmin = fmaxf(fminf(l1, l2), tmin); tmax = fminf(fmaxf(l1, l2), tmax);
    l1 = (BBMin.z - RayOrg.z) * RayDirInv.z; l2 = (BBMax.z - RayOrg.z) * RayDirInv.z;
    tmin = fmaxf(fminf(l1, l2), tmin); tmax = fminf(fmaxf(l1, l2), tmax);
    return ((tmax >= tmin) && (tmax >= 0.0f));
}

__device__ float RayTriangleIntersection(const Ray &r, const float3 &v0,
                                         const float3 &edge1, const float3 &edge2) {
    float3 tvec = r.ori - v0;
    float3 pvec = cross(r.dir, edge2);
    float det = dot(edge1, pvec);
    det = __fdividef(1.0f, det);
    float u = dot(tvec, pvec) * det;
    if (u < 0.0f || u > 1.0f) return -1.0f;
    float3 qvec = cross(tvec, edge1);
    float v = dot(r.dir, qvec) * det;
    if (v < 0.0f || (u + v) > 1.0f) return -1.0f;
    return dot(edge2, qvec) * det;
}

__device__ int RaySphereIntersection(const Ray &ray, const float3 sphere_center,
                                     const float sphere_radius, float &t) {
    float3 sr = ray.ori - sphere_center;
    float b = dot(sr, ray.dir);
    float c = dot(sr, sr) - (sphere_radius * sphere_radius);
    float d = b*b - c;
    if (d > 0) {
        float e = sqrtf(d);
        float t0 = -b - e;
        t = (t0 < 0) ? (-b + e) : fminf(-b - e, -b + e);
        return 1;
    }
    return 0;
}

__global__ void raytrace(unsigned int *out_data, const int w, const int h,
                         const int number_of_triangles, const float4 *tris,
                         const float3 a, const float3 b, const float3 c,
                         const float3 campos, const float3 light_pos, const float3 light_color,
                         const float3 scene_aabb_min, const float3 scene_aabb_max) {
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    float color_x = 0, color_y = 0, color_z = 0;
    int i = 0;

    float randX[] = {0.5f, 0.7f, 0.4f, -1.2f, 0, -.5f, -.2f, 0.25f};
    float randY[] = {0.5f, 0.2f, -.3f,  0.3f, 0, -.4f, -.2f, 0.25f};
    int endIndex = 8;

    for (i = 0; i < endIndex; i++) {
        float xf = (x - randX[i]) / ((float)w);
        float yf = (y - randY[i]) / ((float)h);
        int ray_depth = 0; bool continue_path = true;
        float3 image_pos = (c + (a * xf)) + (b * yf);
        Ray r(image_pos, image_pos - campos);
        HitRecord hit_r;
        float t_min, t_max;
        continue_path = RayBoxIntersection(scene_aabb_min, scene_aabb_max, r.ori, r.inv_dir, t_min, t_max);
        hit_r.color = make_float3(0,0,0);

        float sphere_t;
        bool sphere_hit = RaySphereIntersection(r, light_pos, 2.0f, sphere_t);
        if (sphere_hit && sphere_t > 0.001f) {
            if (!continue_path) hit_r.color = light_color;
            sphere_hit = true;
        }

        while (continue_path && ray_depth < 4) {
            for (int j = 0; j < number_of_triangles; j++) {
                float4 v0 = tris[j*3], e1 = tris[j*3+1], e2 = tris[j*3+2];
                float t = RayTriangleIntersection(r, make_float3(v0.x,v0.y,v0.z),
                            make_float3(e1.x,e1.y,e1.z), make_float3(e2.x,e2.y,e2.z));
                if (t < hit_r.t && t > 0.001f) { hit_r.t = t; hit_r.hit_index = j; }
            }
            if (sphere_hit && sphere_t < hit_r.t) { hit_r.color += light_color; continue_path = false; break; }
            if (hit_r.hit_index >= 0) {
                ray_depth++;
                float4 e1 = tris[hit_r.hit_index*3+1], e2 = tris[hit_r.hit_index*3+2];
                hit_r.normal = normalize(cross(make_float3(e1.x,e1.y,e1.z), make_float3(e2.x,e2.y,e2.z)));
                float3 hitpoint = r.ori + r.dir * hit_r.t;
                float3 L = light_pos - hitpoint;
                float dist_to_light = length(L);
                L = normalize(L);
                float diffuse_light = fminf(fmaxf(dot(L, hit_r.normal), 0.0f), 1.0f);
                float3 H = normalize(L + (-r.dir));
                float specular_light = powf(fmaxf(dot(H, hit_r.normal), 0.0f), 25.0f);
                diffuse_light  *= 16.0f / dist_to_light;
                specular_light *= 16.0f / dist_to_light;
                hit_r.color += light_color * diffuse_light + make_float3(1,1,1)*specular_light*0.2f + make_float3(0.2f,0.2f,0.2f);
                Ray shadow_ray(hitpoint, L);
                for (int j = 0; j < number_of_triangles; j++) {
                    float4 v0 = tris[j*3], e1s = tris[j*3+1], e2s = tris[j*3+2];
                    float t = RayTriangleIntersection(shadow_ray, make_float3(v0.x,v0.y,v0.z),
                                make_float3(e1s.x,e1s.y,e1s.z), make_float3(e2s.x,e2s.y,e2s.z));
                    if (t > 0.025f) { hit_r.color *= 0.25f; break; }
                }
                if (e1.w > 0) { hit_r.resetT(); r = Ray(hitpoint, reflect(r.dir, hit_r.normal)); }
                else continue_path = false;
            } else {
                continue_path = false;
                hit_r.color += make_float3(0.5f, 0.5f, 0.95f*yf + 0.3f);
            }
        }

        if (ray_depth >= 1 || sphere_hit) { ray_depth = max(ray_depth, 1); hit_r.color /= (float)ray_depth; }
        else hit_r.color = make_float3(0.5f, 0.5f, yf + 0.3f);

        color_x += hit_r.color.x; color_y += hit_r.color.y; color_z += hit_r.color.z;
    }

    color_x /= (float)i; color_y /= (float)i; color_z /= (float)i;
    out_data[y * w + x] = rgbToInt(color_x*255, color_y*255, color_z*255);
}

// Playground contract: tris is 3 float4 per triangle (passed as float*), out is
// a w*h packed-RGB image; camera/light/aabb are precomputed by the harness.
extern "C" void solution(const float* tris, unsigned int* out, int w, int h, int n_tris,
                         float3 a, float3 b, float3 c, float3 campos,
                         float3 light_pos, float3 light_color,
                         float3 aabb_min, float3 aabb_max) {
    dim3 block(8, 8, 1);
    dim3 grid(w / block.x, h / block.y, 1);
    raytrace<<<grid, block>>>(out, w, h, n_tris, reinterpret_cast<const float4*>(tris),
                              a, b, c, campos, light_pos, light_color, aabb_min, aabb_max);
}
