import torch
import numpy as np
import argparse

from typing import Dict

DEBUG_PRINT = False

MAX_TENSOR = 6

MAX_TENSOR_DIM = 5

MAX_TENSOR_SIZE = 2**20

DEBUG_TENSOR = False

DEVICE = "cuda"

DTYPE = torch.float

GRAPH_FACTOR = 2

class WrongResultException(Exception):
    pass

def get_broadcast_compatible_shape(tensor_shape):
    max_dim = len(tensor_shape)
    num_b_dims = np.random.randint(0, max_dim + 1)
    trim_head = np.random.randint(0, min(num_b_dims + 1, max_dim))

    shape = tensor_shape[trim_head:max_dim]
    for i in np.random.choice(range(max_dim - trim_head),
                              num_b_dims - trim_head,
                              replace=False):
        shape[i] = 1
    return shape

def random_topology_test(seed, *inp_tensor_list):
    np.random.seed(int(seed.numpy().tolist()))
    tensor_list = [*inp_tensor_list]
    num_tensor = len(tensor_list)

    num_const = np.random.randint(0, num_tensor + 1)
    const_list = np.random.random(num_const)

    if DEBUG_PRINT:
        for const_item in const_list:
            print("----- real number {:.10f}", const_item)

    def get_root(x, dependency_map):
        if x in dependency_map:
            return get_root(dependency_map[x], dependency_map)
        else:
            return x
    d_map: Dict[int, int] = {}
    num_sets = num_tensor
    candidate = list(range(num_tensor))

    unary_operations = [torch.sigmoid, torch.relu]
    binary_operations = [torch.add, torch.sub, torch.mul]
    u_op_size = len(unary_operations)
    b_op_size = len(binary_operations)

    num_operations = np.random.randint(num_sets - 1,
                                       num_sets * GRAPH_FACTOR)

    ret_list = []

    while num_operations >= 0 or num_sets > 1:

        index = np.random.randint(0, len(candidate))
        op_index = np.random.randint(0, u_op_size + b_op_size)
        lh_index = candidate[index]
        rh_index = None
        out_tensor = None

        if DEBUG_PRINT:
            print("iteration {0}, num_sets{1}, candidates {2}, tensor_list {3}, lh_index {4}, op_index {5}".format(
                num_operations, num_sets, candidate, len(tensor_list), lh_index, op_index))
        if num_operations >= 0:
            num_operations -= 1
            if op_index < u_op_size:

                out_tensor = unary_operations[op_index](tensor_list[lh_index])
            else:

                op_2_index = np.random.randint(0, len(tensor_list) + num_const)

                if op_2_index < len(tensor_list):
                    if op_2_index == lh_index:

                        op_2_index = (op_2_index + 1) % len(tensor_list)

                    rh_index = op_2_index
                else:
                    left = tensor_list[lh_index]
                    right = const_list[op_2_index - len(tensor_list)]

                    out_tensor = binary_operations[op_index - u_op_size](left, right)
                if DEBUG_PRINT:
                    print("binary, op_2_index {0}, rh_index ?{1}".format(op_2_index, rh_index))
        else:

            cand_index = np.random.randint(0, len(candidate))
            if cand_index == index:
                cand_index = (cand_index + 1) % len(candidate)

            rh_index = candidate[cand_index]
            if DEBUG_PRINT:
                print("binary rh_index ?{0}".format(rh_index))

        candidate[index] = len(tensor_list)
        lh_root = get_root(lh_index, d_map)

        if rh_index is not None:

            out_tensor = binary_operations[op_index - u_op_size](
                tensor_list[lh_index],
                tensor_list[rh_index])

            if rh_index in candidate:

                candidate.remove(rh_index)

            rh_root = get_root(rh_index, d_map)
            if lh_root != rh_root:
                num_sets -= 1

                d_map[rh_root] = len(tensor_list)

        d_map[lh_root] = len(tensor_list)

        tensor_list.append(out_tensor)

    for ind in candidate:
        ret_list.append(tensor_list[ind])

    out_list = np.random.choice(
        range(num_tensor, len(tensor_list)),
        np.random.randint(0, len(tensor_list) - num_tensor),
        False)
    for ind in out_list:
        if ind not in candidate:
            ret_list.append(tensor_list[ind])

    if DEBUG_PRINT:
        print("ended with tensor_list: {0}".format(len(tensor_list)))

    return tuple(ret_list)

def prepareInputTensorsToRandomTopoTest(seed,
                                        max_tensor_num,
                                        max_tensor_dim,
                                        max_tensor_size,
                                        debug_tensor,
                                        device,
                                        dtype):

    np.random.seed(seed)
    torch.manual_seed(np.random.randint(0, seed))

    seed_tensor = torch.tensor(np.random.randint(0, seed))

    num_tensor = np.random.randint(1, max_tensor_num)

    tensor_dim = np.random.randint(1, max_tensor_dim)
    tensor_shape = []
    numel = 1
    if debug_tensor:
        tensor_shape.append(1)
    else:
        for i in range(tensor_dim):
            size_i = np.random.randint(1, int(max_tensor_size / numel / (2**(tensor_dim - i))))
            size_i = min(size_i, 128 + size_i % 128)
            tensor_shape.insert(0, size_i)
            numel *= size_i

    if DEBUG_PRINT:
        print("output tensor shape: ", tensor_shape)

    num_broadcasted_tensors = np.random.randint(0, 1)

    broadcasted_tensors_indices = np.random.choice(torch.arange(num_tensor),
                                                   num_broadcasted_tensors,
                                                   replace=False)

    tensor_list = []
    for i in range(num_tensor):
        if i in broadcasted_tensors_indices:

            compatible_shape = get_broadcast_compatible_shape(tensor_shape)
            tensor_list.append(torch.randn(compatible_shape, device=device, dtype=dtype) * 100)
        else:
            tensor_list.append(torch.randn(tensor_shape, device=device, dtype=dtype) * 100)
    return seed_tensor, tensor_list

def reproString(current_seed, args):
    repro_str = "python {0}".format(__file__)
    if args.cuda_fuser:
        repro_str += " --cuda_fuser"
    if args.legacy_fuser:
        repro_str += " --legacy_fuser"
    if args.profiling_executor:
        repro_str += " --profiling_executor"
    if args.fp16:
        repro_str += " --fp16"
    if args.cpu:
        repro_str += " --cpu"
    repro_str += " --max_num_tensor {0} --max_tensor_dim {1} --max_tensor_size {2}"\
        " --depth_factor {3} --seed {4} --repro_run".format(
            args.max_num_tensor, args.max_tensor_dim, args.max_tensor_size,
            args.depth_factor, current_seed)
    return repro_str

def runDefaultTestWithSeed(seed):

    seed_tensor, tensor_list = prepareInputTensorsToRandomTopoTest(seed,
                                                                   MAX_TENSOR,
                                                                   MAX_TENSOR_DIM,
                                                                   MAX_TENSOR_SIZE,
                                                                   DEBUG_TENSOR,
                                                                   DEVICE,
                                                                   DTYPE)
    o = random_topology_test(seed_tensor, *tensor_list)
    traced_model = torch.jit.trace(random_topology_test, (seed_tensor, *tensor_list))
    jit_o = traced_model(seed_tensor, *tensor_list)
    jit_o = traced_model(seed_tensor, *tensor_list)
    validate_o = zip(o, jit_o)
    for oo, jit_oo in validate_o:
        if not oo.allclose(jit_oo, atol=1e-5, equal_nan=True):
            return False
    return True

def runTest(seed, args):

    seed_tensor, tensor_list = prepareInputTensorsToRandomTopoTest(seed,
                                                                   args.max_num_tensor,
                                                                   args.max_tensor_dim,
                                                                   args.max_tensor_size,
                                                                   args.debug_tensor,
                                                                   "cuda" if not args.cpu else "cpu",
                                                                   torch.float32 if not args.fp16 else torch.float16)

    try:
        if DEBUG_PRINT:
            print("seed tensor: ", seed_tensor)
        o = random_topology_test(seed_tensor, *tensor_list)
        if DEBUG_PRINT:
            for out in o:
                print("val size: ", out.size())
    except Exception as err:
        raise Exception("Testing script failure with error message, repro by running:\n"
                        f"\t{reproString(seed, args)}") from err
    try:
        traced_model = torch.jit.trace(random_topology_test, (seed_tensor, *tensor_list))
        if DEBUG_PRINT:
            print("original graph: ", traced_model.graph)
        jit_o = traced_model(seed_tensor, *tensor_list)
        jit_o = traced_model(seed_tensor, *tensor_list)
        if DEBUG_PRINT:
            print("optimized graph: ", traced_model.graph_for(seed_tensor, *tensor_list))

        validate_o = zip(o, jit_o)
        for oo, jit_oo in validate_o:
            if not oo.allclose(jit_oo, equal_nan=True):
                print("eager output: ", oo)
                print("jit output: ", jit_oo)
                print("diff ", jit_oo - oo)
                raise WrongResultException()
    except WrongResultException as err:
        raise Exception("cuda fuser gives wrong results, repro by running:\n"
                        f"\t{reproString(seed, args)}") from err
    except Exception as err:
        raise Exception("something in cuda fuser went wrong, repro by running:\n"
                        f"\t{reproString(seed, args)}") from err

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda_fuser", action='store_true', default=True)
    parser.add_argument("--legacy_fuser", action='store_true', default=False)
    parser.add_argument("--profiling_executor", action='store_true', default=False)
    parser.add_argument("--fp16", action='store_true', default=False)
    parser.add_argument("--cpu", action='store_true', default=False)
    parser.add_argument("--debug_print", action='store_true', default=False)
    parser.add_argument("--debug_tensor", action='store_true', default=False)
    parser.add_argument("--max_num_tensor", default=MAX_TENSOR, type=int)
    parser.add_argument("--max_tensor_dim", default=MAX_TENSOR_DIM, type=int)
    parser.add_argument("--max_tensor_size", default=MAX_TENSOR_SIZE, type=int)
    parser.add_argument("--depth_factor", default=GRAPH_FACTOR, type=int)
    parser.add_argument("--seed", default=45589, type=int)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--iterations", default=4, type=int)
    group.add_argument("--repro_run", action='store_true', default=False)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    if args.cuda_fuser:
        torch._C._jit_set_nvfuser_enabled(True)

    if not args.legacy_fuser:
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_override_can_fuse_on_gpu(False)

    if not args.profiling_executor:
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_set_profiling_mode(False)

    GRAPH_FACTOR = args.depth_factor

    DEBUG_PRINT = args.debug_print

    if args.repro_run:
        runTest(args.seed, args)
    else:
        np.random.seed(args.seed)
        failing_repros = []
        for seed in np.random.randint(0, args.seed, args.iterations):
            try:
                runTest(seed, args)
            except Exception as e:
                failing_repros.append(str(e))
        if len(failing_repros) == 0:
            print("test passed")
        else:
            print("{0} out of {1} tests failed;".format(
                  len(failing_repros), args.iterations))
            print("To repro failing tests, run\n")
            for repro in failing_repros:
                print(repro)
