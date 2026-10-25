// This shader draws a set of splats, each defined by a center and an extent

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

struct Splat {
    center: vec2<f32>,
    conic: vec3<f32>,
    color: vec4<f32>,
    radius: f32,
    extent: vec2<f32>,
};

@group(0) @binding(0)
var<storage, read> splats: array<Splat>;
@group(0) @binding(1)
var<storage, read> sorted_indices: array<u32>;

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
    var out: VertexOutput;
    let corners = array<vec2<f32>,4>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0,  1.0),
    );
    let splat = splats[sorted_indices[instance_index]];
    let ndc = splat.center + splat.extent * corners[vertex_index % 4u];
    out.position = vec4<f32>(ndc, 0.0, 1.0);
    out.color = splat.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color.rgb, 1.0);
}
