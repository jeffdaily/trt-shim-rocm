#include <cstdio>
#include <vector>
#include <migraphx/migraphx.hpp>
int main(int argc, char** argv) {
    std::FILE* f = std::fopen(argv[1], "rb");
    std::fseek(f, 0, SEEK_END); long n = std::ftell(f); std::fseek(f, 0, SEEK_SET);
    std::vector<char> buf(n); fread(buf.data(),1,n,f); fclose(f);
    migraphx::onnx_options o;
    migraphx::dynamic_dimension dd(1, 4);  // batch min=1 max=4
    o.set_default_dyn_dim_value(dd);
    auto prog = migraphx::parse_onnx_buffer(buf.data(), buf.size(), o);
    migraphx::compile_options c; c.set_offload_copy(true);
    prog.compile(migraphx::target("gpu"), c);
    auto ps = prog.get_parameter_shapes();
    for (auto name : ps.names()) {
        auto s = ps[name];
        printf("param %s: dynamic=%d ndim try...\n", name, (int)s.dynamic());
        auto L = s.lengths();
        printf("  lengths.size=%zu: ", L.size());
        for (auto v : L) printf("%zu ", v);
        printf("\n");
    }
    // eval at batch 2
    std::vector<float> in(2*3*224*224, 0.1f);
    migraphx::shape ins(migraphx_shape_float_type, {2,3,224,224});
    migraphx::program_parameters pp;
    pp.add(ps.names()[0], migraphx::argument(ins, in.data()));
    auto res = prog.eval(pp);
    auto os = res[0].get_shape();
    auto ol = os.lengths();
    printf("output ndim=%zu: ", ol.size());
    for (auto v : ol) printf("%zu ", v);
    printf("\n");
    return 0;
}
