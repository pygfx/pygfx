from math import ceil

import numpy as np

from pygfx.resources._utils import (
    get_alignment_multiplier,
    calculate_buffer_chunk_size,
    calculate_texture_chunk_size,
    get_merged_blocks_from_mask_3d,
)


def calculate_and_show_chunk_size(tex_size, target_chunk_count=20, **kwargs):
    """Function to print stats on the chunks. Helpful during dev."""
    chunk_size = calculate_texture_chunk_size(
        tex_size, target_chunk_count=target_chunk_count, **kwargs
    )

    nchunks_list = [ts / cs for ts, cs in zip(tex_size, chunk_size)]
    nchunks = ceil(nchunks_list[0]) * ceil(nchunks_list[1]) * ceil(nchunks_list[2])
    chunk_counts_str = ", ".join(f"{x:0.03f}" for x in nchunks_list)
    print(f"{tex_size} -> {chunk_size} - {nchunks} chunks (counts: {chunk_counts_str})")
    return chunk_size


def test_get_alignment_multiplier():
    for bpe, align, result in [
        (1, 1, 1),
        (2, 1, 1),
        (3, 1, 1),
        (4, 1, 1),
        (16, 1, 1),
        #
        (1, 16, 16),
        (2, 16, 8),
        (3, 16, 16),
        (4, 16, 4),
        (5, 16, 16),
        (6, 16, 8),
        (7, 16, 16),
        (8, 16, 2),
        (9, 16, 16),
        (10, 16, 8),
        (11, 16, 16),
        (12, 16, 4),
        (13, 16, 16),
        (14, 16, 8),
        (15, 16, 16),
        (16, 16, 1),
        #
        (17, 16, 16),
        (18, 16, 8),
        (24, 16, 2),
        (32, 16, 1),
        #
        (1, 32, 32),
        (2, 32, 16),
        (4, 32, 8),
        (5, 32, 32),
        (8, 32, 4),
        (12, 32, 8),
        (16, 32, 2),
        (32, 32, 1),
    ]:
        assert get_alignment_multiplier(bpe, align) == result


def test_calculate_buffer_chunk_size():
    # We basically test this in all dim1 tests below. But test a bit for good measure

    assert calculate_buffer_chunk_size(10) == 10
    assert calculate_buffer_chunk_size(100) == 100
    assert calculate_buffer_chunk_size(100, min_chunk_bytes=10) == 16

    assert calculate_buffer_chunk_size(1000, target_chunk_count=20) == 256
    assert calculate_buffer_chunk_size(10000, target_chunk_count=20) == 512

    assert calculate_buffer_chunk_size(12345, target_chunk_count=20) == 624
    assert (
        calculate_buffer_chunk_size(12345, target_chunk_count=20, byte_align=1) == 618
    )
    assert (
        calculate_buffer_chunk_size(12345, target_chunk_count=20, byte_align=256) == 768
    )


def test_chunk_size_dim1_bounds():
    # Note the chunk size bounds 256
    assert calculate_and_show_chunk_size((10, 1, 1)) == (10, 1, 1)
    assert calculate_and_show_chunk_size((100, 1, 1)) == (100, 1, 1)
    assert calculate_and_show_chunk_size((1_000, 1, 1)) == (256, 1, 1)

    # Now it scales nicely
    assert calculate_and_show_chunk_size((10_000, 1, 1)) == (512, 1, 1)
    assert calculate_and_show_chunk_size((100_000, 1, 1)) == (5008, 1, 1)
    assert calculate_and_show_chunk_size((1_000_000, 1, 1)) == (50000, 1, 1)
    assert calculate_and_show_chunk_size((10_000_000, 1, 1)) == (500000, 1, 1)

    # And here it gets stuck at about 1MB
    assert calculate_and_show_chunk_size((100_000_000, 1, 1)) == (1041680, 1, 1)
    assert calculate_and_show_chunk_size((1_000_000_000, 1, 1)) == (1048224, 1, 1)

    # Chunk size bounds are expressed in bytes
    assert calculate_and_show_chunk_size(
        (1_000_000_000, 1, 1), bytes_per_element=2
    ) == (524112, 1, 1)
    assert calculate_and_show_chunk_size(
        (1_000_000_000, 1, 1), bytes_per_element=8
    ) == (131062, 1, 1)

    # Check giving min and max chunk size (the lower bound is approximate)
    assert calculate_and_show_chunk_size((10_000, 1, 1), min_chunk_bytes=2**13) == (
        5008,
        1,
        1,
    )
    assert calculate_and_show_chunk_size((100_000, 1, 1), max_chunk_bytes=512) == (
        512,
        1,
        1,
    )


def test_chunk_size_dim1_coverage():
    # The algorithm tries to avoid half-chunks at the end

    # 20_160 / 1008 -> 20.0 chunks
    assert calculate_and_show_chunk_size((20_160, 1, 1)) == (1008, 1, 1)
    # 20_161 / 1008 -> 20.001 chunks -> not good, would need extra chunk for that bit of data
    # 20_161 / 1024 -> 19.69 chunks -> better use of space
    assert calculate_and_show_chunk_size((20_161, 1, 1)) == (1024, 1, 1)


def test_chunk_size_dim1_align():
    res = calculate_and_show_chunk_size((10_123, 1, 1), bytes_per_element=1)
    assert res == (512, 1, 1)
    res = calculate_and_show_chunk_size((10_123, 1, 1), bytes_per_element=3)
    assert res == (512, 1, 1)
    res = calculate_and_show_chunk_size((10_123, 1, 1), bytes_per_element=33)
    assert res == (512, 1, 1)
    res = calculate_and_show_chunk_size((10_123, 1, 1), bytes_per_element=2)
    assert res == (512, 1, 1)
    res = calculate_and_show_chunk_size((10_123, 1, 1), bytes_per_element=4)
    assert res == (508, 1, 1)
    res = calculate_and_show_chunk_size((10_123, 1, 1), bytes_per_element=8)
    assert res == (508, 1, 1)
    res = calculate_and_show_chunk_size((10_123, 1, 1), bytes_per_element=16)
    assert res == (507, 1, 1)
    res = calculate_and_show_chunk_size((10_123, 1, 1), bytes_per_element=32)
    assert res == (507, 1, 1)
    res = calculate_and_show_chunk_size(
        (10_123, 1, 1), bytes_per_element=16, byte_align=64
    )
    assert res == (508, 1, 1)


def test_chunk_size_dim2_bounds():
    # Below lower bound
    assert calculate_and_show_chunk_size((10, 10, 1)) == (10, 10, 1)
    assert calculate_and_show_chunk_size((20, 20, 1)) == (20, 10, 1)
    # Some scale cane be seen
    assert calculate_and_show_chunk_size((100, 100, 1)) == (100, 5, 1)
    assert calculate_and_show_chunk_size((1_000, 1_000, 1)) == (256, 50, 1)

    # Upper bound kicks in
    assert calculate_and_show_chunk_size((10_000, 10_000, 1)) == (512, 500, 1)
    assert calculate_and_show_chunk_size((100_000, 100_000, 1)) == (5008, 210, 1)

    # Max size is in bytes, and is affected by bytes_per_element as well
    res = calculate_and_show_chunk_size((100_000, 100_000, 1), max_chunk_bytes=2**19)
    assert res == (5008, 105, 1)
    res = calculate_and_show_chunk_size((100_000, 100_000, 1), bytes_per_element=2)
    assert res == (5000, 105, 1)


def test_chunk_size_dim2_ratio():
    # Chunk sizes are more or less equal.
    # A ratio of 2 or so is to be expected due to rounding and alignment.
    assert calculate_and_show_chunk_size((1000, 2000, 1), bytes_per_element=64) == (
        50,
        100,
        1,
    )
    assert calculate_and_show_chunk_size((1000, 10000, 1), bytes_per_element=64) == (
        50,
        323,
        1,
    )
    assert calculate_and_show_chunk_size((2000, 1000, 1), bytes_per_element=64) == (
        100,
        50,
        1,
    )
    assert calculate_and_show_chunk_size((10000, 1000, 1), bytes_per_element=64) == (
        500,
        33,
        1,
    )


def test_chunk_size_dim2_align():
    # The first dimension is aligned, the second is not
    res = calculate_and_show_chunk_size((1234, 876, 1), bytes_per_element=1)
    assert res == (256, 44, 1)
    res = calculate_and_show_chunk_size((1234, 876, 1), bytes_per_element=3)
    assert res == (96, 44, 1)
    res = calculate_and_show_chunk_size((1234, 876, 1), bytes_per_element=2)
    assert res == (128, 44, 1)
    res = calculate_and_show_chunk_size((1234, 876, 1), bytes_per_element=4)
    assert res == (64, 44, 1)
    res = calculate_and_show_chunk_size((1234, 876, 1), bytes_per_element=8)
    assert res == (62, 44, 1)
    res = calculate_and_show_chunk_size((1234, 876, 1), bytes_per_element=16)
    assert res == (62, 44, 1)


def test_chunk_size_dim3_bounds():
    # Below lower bound
    assert calculate_and_show_chunk_size((4, 4, 4)) == (4, 4, 4)
    assert calculate_and_show_chunk_size((10, 10, 10)) == (10, 10, 2)
    # Some scale can be seen
    assert calculate_and_show_chunk_size((20, 20, 20)) == (20, 10, 1)
    assert calculate_and_show_chunk_size((100, 100, 100)) == (100, 5, 5)

    # Upper bound kicks in
    assert calculate_and_show_chunk_size((1000, 1000, 1000)) == (1000, 50, 21)
    assert calculate_and_show_chunk_size((2000, 2000, 2000)) == (2000, 100, 6)

    # Max size is in bytes, and is affected by bytes_per_element as well
    res = calculate_and_show_chunk_size((2000, 2000, 2000), max_chunk_bytes=2**19)
    assert res == (2000, 100, 3)
    res = calculate_and_show_chunk_size((2000, 2000, 2000), bytes_per_element=2)
    assert res == (2000, 100, 3)


def test_chunk_size_dim3_ratio():
    # Chunk sizes are more or less equal.
    # A ratio of 2 or so is to be expected due to rounding and alignment.
    # The first dim is always full, in 3d
    assert calculate_and_show_chunk_size((100, 200, 300), bytes_per_element=64) == (
        100,
        10,
        15,
    )
    assert calculate_and_show_chunk_size(
        (100, 200, 300), bytes_per_element=64, target_chunk_count=100
    ) == (
        100,
        2,
        3,
    )
    assert calculate_and_show_chunk_size((1000, 2000, 3000)) == (1000, 100, 11)

    assert calculate_and_show_chunk_size((300, 200, 100)) == (300, 10, 5)
    assert calculate_and_show_chunk_size((300, 200, 100), target_chunk_count=100) == (
        300,
        2,
        1,
    )
    assert calculate_and_show_chunk_size((3000, 2000, 1000)) == (3000, 100, 4)

    assert calculate_and_show_chunk_size((400, 100, 100)) == (400, 5, 5)
    assert calculate_and_show_chunk_size((100, 400, 100)) == (100, 20, 5)
    assert calculate_and_show_chunk_size((100, 100, 400)) == (100, 5, 20)


def test_chunk_size_dim3_align():
    # The first dimension is aligned, the second is not
    # Well, the first is always full anyway ...
    res = calculate_and_show_chunk_size((121, 87, 65), bytes_per_element=1)
    assert res == (121, 5, 4)
    res = calculate_and_show_chunk_size((121, 87, 65), bytes_per_element=3)
    assert res == (121, 5, 4)
    res = calculate_and_show_chunk_size((121, 87, 65), bytes_per_element=2)
    assert res == (121, 5, 4)
    res = calculate_and_show_chunk_size((121, 87, 65), bytes_per_element=4)
    assert res == (121, 5, 4)
    res = calculate_and_show_chunk_size((121, 87, 65), bytes_per_element=8)
    assert res == (121, 5, 4)
    res = calculate_and_show_chunk_size((121, 87, 65), bytes_per_element=16)
    assert res == (121, 5, 4)


def make_mask_3d(mask):
    mask = np.array(mask, bool)
    if mask.ndim == 1:
        mask.shape = 1, 1, mask.shape[0]
    elif mask.ndim == 2:
        mask.shape = 1, mask.shape[0], mask.shape[1]
    return mask


def test_chunk_merging_1d():
    # One block
    mask = [0, 1, 0, 0, 0, 0, 0]
    blocks = get_merged_blocks_from_mask_3d(make_mask_3d(mask))
    assert len(blocks) == 1
    assert blocks[0].get_offset() == (1, 0, 0)
    assert blocks[0].get_size() == (1, 1, 1)

    # Merged blocks
    mask = [0, 0, 1, 1, 1, 0, 0]
    blocks = get_merged_blocks_from_mask_3d(make_mask_3d(mask))
    assert len(blocks) == 1
    assert blocks[0].get_offset() == (2, 0, 0)
    assert blocks[0].get_size() == (3, 1, 1)

    # Merging is aggressive
    mask = [0, 0, 1, 1, 0, 1, 0]
    blocks = get_merged_blocks_from_mask_3d(make_mask_3d(mask))
    assert len(blocks) == 2
    assert blocks[0].get_offset() == (2, 0, 0)
    assert blocks[0].get_size() == (2, 1, 1)
    assert blocks[1].get_offset() == (5, 0, 0)
    assert blocks[1].get_size() == (1, 1, 1)

    # Two blocks
    mask = [1, 0, 0, 1, 1, 1, 1]
    blocks = get_merged_blocks_from_mask_3d(make_mask_3d(mask))
    assert len(blocks) == 2
    assert blocks[0].get_offset() == (0, 0, 0)
    assert blocks[0].get_size() == (1, 1, 1)
    assert blocks[1].get_offset() == (3, 0, 0)
    assert blocks[1].get_size() == (4, 1, 1)


def test_chunk_merging_2d():
    # One block
    mask = [
        [0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]
    blocks = get_merged_blocks_from_mask_3d(make_mask_3d(mask))
    assert len(blocks) == 1
    assert blocks[0].get_offset() == (1, 1, 0)
    assert blocks[0].get_size() == (1, 2, 1)

    # One larger block
    mask = [
        [0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0],
        [0, 1, 1, 0, 0, 0],
        [0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]
    blocks = get_merged_blocks_from_mask_3d(make_mask_3d(mask))
    assert len(blocks) == 1
    assert blocks[0].get_offset() == (0, 1, 0)
    assert blocks[0].get_size() == (6, 3, 1)

    # Aggressive merging in x
    mask = [
        [0, 0, 0, 0, 0, 0],
        [1, 0, 1, 1, 0, 0],
        [1, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]
    blocks = get_merged_blocks_from_mask_3d(make_mask_3d(mask))
    assert len(blocks) == 1
    assert blocks[0].get_offset() == (0, 1, 0)
    assert blocks[0].get_size() == (6, 2, 1)

    # Normal merging in y
    mask = [
        [0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]
    blocks = get_merged_blocks_from_mask_3d(make_mask_3d(mask))
    assert len(blocks) == 2
    assert blocks[0].get_offset() == (0, 1, 0)
    assert blocks[0].get_size() == (6, 1, 1)
    assert blocks[1].get_offset() == (0, 3, 0)
    assert blocks[1].get_size() == (6, 1, 1)

    # Merging has some leeway
    mask = [
        [0, 0, 0, 0, 0, 0],
        [1, 0, 1, 1, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]
    blocks = get_merged_blocks_from_mask_3d(make_mask_3d(mask))
    assert len(blocks) == 1
    assert blocks[0].get_offset() == (0, 1, 0)
    assert blocks[0].get_size() == (6, 2, 1)

    # Dito
    mask = [
        [0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]
    blocks = get_merged_blocks_from_mask_3d(make_mask_3d(mask))
    assert len(blocks) == 1
    assert blocks[0].get_offset() == (0, 1, 0)
    assert blocks[0].get_size() == (6, 4, 1)

    # Three blocks
    mask = [
        [1, 1, 0, 0, 1, 1, 1],
        [1, 1, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 0],
    ]
    blocks = get_merged_blocks_from_mask_3d(make_mask_3d(mask))
    assert len(blocks) == 3
    assert blocks[0].get_offset() == (0, 0, 0)
    assert blocks[0].get_size() == (7, 2, 1)
    assert blocks[1].get_offset() == (0, 3, 0)
    assert blocks[1].get_size() == (7, 1, 1)
    assert blocks[2].get_offset() == (0, 4, 0)
    assert blocks[2].get_size() == (7, 2, 1)


def test_chunk_merging_3d():
    # One small block
    mask = [
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        [
            [0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        [
            [0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
    ]
    blocks = get_merged_blocks_from_mask_3d(make_mask_3d(mask))
    assert len(blocks) == 1
    assert blocks[0].get_offset() == (1, 0, 1)
    assert blocks[0].get_size() == (1, 4, 2)

    # One larger block, plus aggressive merging in x
    mask = [
        [
            [0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 1, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        [
            [0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 1, 0],
            [0, 1, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        [
            [0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 1, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
    ]
    blocks = get_merged_blocks_from_mask_3d(make_mask_3d(mask))
    assert len(blocks) == 1
    assert blocks[0].get_offset() == (0, 0, 0)
    assert blocks[0].get_size() == (6, 4, 3)

    # Flip one element to True, extending the block
    mask[0][3][3] = 1
    blocks = get_merged_blocks_from_mask_3d(make_mask_3d(mask))
    assert len(blocks) == 1
    assert blocks[0].get_offset() == (0, 0, 0)
    assert blocks[0].get_size() == (6, 4, 3)

    # Two blocks
    mask = [
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 1, 1],
        ],
        [
            [0, 1, 1, 1, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 1, 1],
        ],
        [
            [0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 1, 1],
        ],
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 1, 1],
        ],
    ]
    blocks = get_merged_blocks_from_mask_3d(make_mask_3d(mask))
    assert len(blocks) == 2
    assert blocks[0].get_offset() == (0, 0, 0)
    assert blocks[0].get_size() == (8, 4, 3)
    assert blocks[1].get_offset() == (0, 0, 3)
    assert blocks[1].get_size() == (8, 4, 1)


if __name__ == "__main__":
    test_calculate_buffer_chunk_size()

    test_chunk_size_dim1_bounds()
    test_chunk_size_dim1_coverage()
    test_chunk_size_dim1_align()

    test_chunk_size_dim2_bounds()
    test_chunk_size_dim2_ratio()
    test_chunk_size_dim2_align()

    test_chunk_size_dim3_bounds()
    test_chunk_size_dim3_ratio()
    test_chunk_size_dim3_align()

    test_chunk_merging_1d()
    test_chunk_merging_2d()
    test_chunk_merging_3d()
