"""Smoke tests for the native texture-generation extensions.

Run ``./build_extensions.sh`` before executing this test module.
"""

import unittest

import torch

import custom_rasterizer_kernel
from hy3dgen.texgen.differentiable_renderer import mesh_processor


class NativeExtensionTests(unittest.TestCase):
    @staticmethod
    def _rasterize_triangle(device: str) -> tuple[torch.Tensor, torch.Tensor]:
        vertices = torch.tensor(
            [
                [-0.6, -0.6, 0.0, 1.0],
                [0.6, -0.6, 0.0, 1.0],
                [0.0, 0.6, 0.0, 1.0],
            ],
            dtype=torch.float32,
            device=device,
        )
        faces = torch.tensor([[0, 1, 2]], dtype=torch.int32, device=device)
        depth = torch.empty(0, dtype=torch.float32, device=device)

        face_indices, barycentric = custom_rasterizer_kernel.rasterize_image(
            vertices, faces, depth, 32, 32, 1e-6, 0
        )
        if device == "cuda":
            torch.cuda.synchronize()
        return face_indices, barycentric

    def _assert_valid_triangle(self, device: str) -> None:
        face_indices, barycentric = self._rasterize_triangle(device)

        self.assertEqual(face_indices.shape, (32, 32))
        self.assertEqual(barycentric.shape, (32, 32, 3))
        covered = face_indices > 0
        self.assertGreater(int(covered.sum()), 0)
        torch.testing.assert_close(
            barycentric[covered].sum(dim=-1),
            torch.ones(int(covered.sum()), device=device),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_cpu_rasterizer(self) -> None:
        self._assert_valid_triangle("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device is not available")
    def test_cuda_rasterizer(self) -> None:
        self._assert_valid_triangle("cuda")

    def test_mesh_processor_import(self) -> None:
        self.assertTrue(mesh_processor.__file__)


if __name__ == "__main__":
    unittest.main()
