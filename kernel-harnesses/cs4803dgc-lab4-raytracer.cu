#include "../tensor-lib/tensor.cuh"
#include "../kernel-implementation/cutil_math_min.cuh"   // float3 ops (host) for the camera
#include <vector>

// CS4803 Lab 4 — CUDA ray tracer. Renders a small hardcoded scene (floor quad +
// a standing reflective triangle, lit by a point light drawn as a sphere) into a
// w*h packed-RGB image. Camera math replicates lab4's updateCamera().
extern "C" void solution(const float* tris, unsigned int* out, int w, int h, int n_tris,
                         float3 a, float3 b, float3 c, float3 campos,
                         float3 light_pos, float3 light_color,
                         float3 aabb_min, float3 aabb_max);

int main() {
    tensor::begin("cs4803dgc-lab4-raytracer");

    int w = 512, h = 512;   // multiples of 8 (block size)

    // Scene triangles (v0, v1, v2, reflective?). Stored on device as 3 float4
    // each: (v0, edge1=v1-v0 [w=reflective], edge2=v2-v0).
    struct Tri { float3 v0, v1, v2; float refl; };
    std::vector<Tri> mesh = {
        {{-4,0,-4}, { 4,0,-4}, { 4,0, 4}, 0},   // floor
        {{-4,0,-4}, { 4,0, 4}, {-4,0, 4}, 0},
        {{-2,0, 0}, { 2,0, 0}, { 0,3, 0}, 1},   // standing reflective triangle
    };
    int n_tris = (int)mesh.size();

    tensor::Buffer<float> tris(n_tris * 12);
    float3 amin = make_float3(1e9f,1e9f,1e9f), amax = make_float3(-1e9f,-1e9f,-1e9f);
    for (int i = 0; i < n_tris; i++) {
        float3 v0 = mesh[i].v0, e1 = mesh[i].v1 - mesh[i].v0, e2 = mesh[i].v2 - mesh[i].v0;
        float* p = tris.host + i * 12;
        p[0]=v0.x; p[1]=v0.y; p[2]=v0.z; p[3]=0;
        p[4]=e1.x; p[5]=e1.y; p[6]=e1.z; p[7]=mesh[i].refl;
        p[8]=e2.x; p[9]=e2.y; p[10]=e2.z; p[11]=0;
        for (float3 v : {mesh[i].v0, mesh[i].v1, mesh[i].v2}) {
            amin.x=fminf(amin.x,v.x); amin.y=fminf(amin.y,v.y); amin.z=fminf(amin.z,v.z);
            amax.x=fmaxf(amax.x,v.x); amax.y=fmaxf(amax.y,v.y); amax.z=fmaxf(amax.z,v.z);
        }
    }
    cudaMemcpy((float*)tris, tris.host, n_tris * 12 * sizeof(float), cudaMemcpyHostToDevice);

    tensor::Buffer<uint32_t> out(w * h);

    // Point light (also expand the scene AABB to include its sphere).
    float3 light_pos = make_float3(3, 6, 2), light_color = make_float3(1,1,1);
    amin = make_float3(fminf(amin.x,light_pos.x-2), fminf(amin.y,light_pos.y-2), fminf(amin.z,light_pos.z-2));
    amax = make_float3(fmaxf(amax.x,light_pos.x+2), fmaxf(amax.y,light_pos.y+2), fmaxf(amax.z,light_pos.z+2));

    // Camera (replicates lab4 updateCamera): 60deg FOV, orbiting eye.
    float cam_rot = 0.7f, cam_dist = 10.0f, cam_height = 6.0f;
    float3 campos = make_float3(cosf(cam_rot)*cam_dist, cam_height, -sinf(cam_rot)*cam_dist);
    float3 cam_dir = normalize(-campos);
    float3 cam_right = normalize(cross(cam_dir, make_float3(0,1,0)));
    float3 cam_up = normalize(-cross(cam_dir, cam_right));
    float theta = (60.0f * 3.1415f * 0.5f) / 180.0f, half_w = tanf(theta), aspect = (float)w/(float)h;
    float u0 = -half_w*aspect, v0 = -half_w, u1 = half_w*aspect, v1 = half_w;
    float3 a = (u1-u0)*cam_right;
    float3 b = (v1-v0)*cam_up;
    float3 c = campos + u0*cam_right + v0*cam_up + 1.0f*cam_dir;

    BENCHMARK(solution(tris, out, w, h, n_tris, a, b, c, campos, light_pos, light_color, amin, amax));

    out.preview("image");
    tensor::end();
}
