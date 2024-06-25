from math import ceil

from pygfx.resources._base import get_alignment_multiplier, calculate_buffer_chunk_size, calculate_texture_chunk_size


def calculate_and_show_chunk_size(tex_size, **kwargs):
    """Function to print stats on the chunks. Helpful during dev.
    """
    chunk_size = calculate_texture_chunk_size(tex_size, **kwargs)

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
    assert calculate_buffer_chunk_size(100, min_chunk_size=10) == 16

    assert calculate_buffer_chunk_size(1000) == 256
    assert calculate_buffer_chunk_size(10000) == 512

    assert calculate_buffer_chunk_size(12345) == 624
    assert calculate_buffer_chunk_size(12345, byte_align=1) == 618
    assert calculate_buffer_chunk_size(12345, byte_align=256) == 768



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
    assert calculate_and_show_chunk_size((1_000_000_000, 1, 1), bytes_per_element=2) == (524112, 1, 1)
    assert calculate_and_show_chunk_size((1_000_000_000, 1, 1), bytes_per_element=8) == (131062, 1, 1)

    # Check giving min and max chunk size (the lower bound is approximate)
    assert calculate_and_show_chunk_size((10_000, 1, 1), min_chunk_size=2**13) == (5008, 1, 1)
    assert calculate_and_show_chunk_size((100_000, 1, 1), max_chunk_size=512) == (512, 1, 1)


def test_chunk_size_dim1_coverage():
    # The algorithm tries to avoid half-chunks at the end

    # 20_160 / 1008 -> 20.0 chunks
    assert calculate_and_show_chunk_size((20_160, 1, 1)) == (1008, 1, 1)
    # 20_161 / 1008 -> 20.001 chunks -> not good, would need extra chunk for that bit of data
    # 20_161 / 1024 -> 19.69 chunks -> better use of space
    assert calculate_and_show_chunk_size((20_161, 1, 1)) == (1024, 1, 1)


def test_chunk_size_dim1_align():
    assert calculate_and_show_chunk_size((10_123, 1, 1), bytes_per_element=1) == (512, 1, 1)
    assert calculate_and_show_chunk_size((10_123, 1, 1), bytes_per_element=3) == (512, 1, 1)
    assert calculate_and_show_chunk_size((10_123, 1, 1), bytes_per_element=33) == (512, 1, 1)
    assert calculate_and_show_chunk_size((10_123, 1, 1), bytes_per_element=2) == (512, 1, 1)
    assert calculate_and_show_chunk_size((10_123, 1, 1), bytes_per_element=4) == (508, 1, 1)
    assert calculate_and_show_chunk_size((10_123, 1, 1), bytes_per_element=8) == (508, 1, 1)
    assert calculate_and_show_chunk_size((10_123, 1, 1), bytes_per_element=16) == (507, 1, 1)
    assert calculate_and_show_chunk_size((10_123, 1, 1), bytes_per_element=32) == (507, 1, 1)
    assert calculate_and_show_chunk_size((10_123, 1, 1), bytes_per_element=16, byte_align=64) == (508, 1, 1)


def test_chunk_size_dim2_bounds():
    # Below lower bound
    assert calculate_and_show_chunk_size((10, 10, 1)) == (10, 10, 1)
    assert calculate_and_show_chunk_size((20, 20, 1)) == (16, 10, 1)
    # Some scale cane be seen
    assert calculate_and_show_chunk_size((100, 100, 1)) == (32, 15, 1)
    assert calculate_and_show_chunk_size((1_000, 1_000, 1)) == (208, 200, 1)

    # Upper bound kicks in
    assert calculate_and_show_chunk_size((10_000, 10_000, 1)) == (1008, 1000, 1)
    assert calculate_and_show_chunk_size((100_000, 100_000, 1)) == (1024, 1021, 1)

    # Max size is in bytes, and is affected by bytes_per_element as well
    assert calculate_and_show_chunk_size((100_000, 100_000, 1), max_chunk_size=2**19) == (720, 725, 1)
    assert calculate_and_show_chunk_size((100_000, 100_000, 1), bytes_per_element=2) == (720, 725, 1)


def test_chunk_size_dim2_ratio():
    # Chunk sizes are more or less equal.
    # A ratio of 2 or so is to be expected due to rounding and alignment.
    assert calculate_and_show_chunk_size((1000, 2000, 1)) == (256, 334, 1)
    assert calculate_and_show_chunk_size((1000, 10000, 1)) == (512, 910, 1)
    assert calculate_and_show_chunk_size((2000, 1000, 1)) == (400, 250, 1)
    assert calculate_and_show_chunk_size((10000, 1000, 1)) == (1008, 500, 1)


def test_chunk_size_dim2_align():
    # The first dimension is aligned, the second is not
    assert calculate_and_show_chunk_size((1234, 876, 1), bytes_per_element=1) == (256, 219, 1)
    assert calculate_and_show_chunk_size((1234, 876, 1), bytes_per_element=3) == (256, 219, 1)
    assert calculate_and_show_chunk_size((1234, 876, 1), bytes_per_element=2) == (248, 219, 1)
    assert calculate_and_show_chunk_size((1234, 876, 1), bytes_per_element=4) == (248, 219, 1)
    assert calculate_and_show_chunk_size((1234, 876, 1), bytes_per_element=8) == (248, 219, 1)
    assert calculate_and_show_chunk_size((1234, 876, 1), bytes_per_element=16) == (247, 219, 1)



def test_chunk_size_dim3_bounds():
    # Below lower bound
    assert calculate_and_show_chunk_size((4, 4, 4)) == (4, 4, 4)
    assert calculate_and_show_chunk_size((10, 10, 10)) == (10, 5, 5)
    # Some scale cane be seen
    assert calculate_and_show_chunk_size((20, 20, 20)) == (16, 5, 5)
    assert calculate_and_show_chunk_size((100, 100, 100)) == (48, 25, 34)

    # Upper bound kicks in
    assert calculate_and_show_chunk_size((1000, 1000, 1000)) == (112, 91, 100)
    assert calculate_and_show_chunk_size((2000, 2000, 2000)) == (112, 96, 96)

    # Max size is in bytes, and is affected by bytes_per_element as well
    assert calculate_and_show_chunk_size((2000, 2000, 2000), max_chunk_size=2**19) == (80, 80, 80)
    assert calculate_and_show_chunk_size((2000, 2000, 2000), bytes_per_element=2) == (80, 80, 80)


def test_chunk_size_dim3_ratio():
    # Chunk sizes are more or less equal.
    # A ratio of 2 or so is to be expected due to rounding and alignment.
    assert calculate_and_show_chunk_size((100, 200, 300)) == (64, 67, 60)
    assert calculate_and_show_chunk_size((100, 200, 300), target_chunk_count=100) == (48, 34, 34)
    assert calculate_and_show_chunk_size((1000, 2000, 3000)) == (112, 96, 97)

    assert calculate_and_show_chunk_size((300, 200, 100)) == (80, 67, 50)
    assert calculate_and_show_chunk_size((300, 200, 100), target_chunk_count=100) == (48, 40, 34)
    assert calculate_and_show_chunk_size((3000, 2000, 1000)) == (112, 100, 100)

    assert calculate_and_show_chunk_size((400, 100, 100)) == (80, 50, 50)
    assert calculate_and_show_chunk_size((100, 400, 100)) == (64, 58, 50)
    assert calculate_and_show_chunk_size((100, 100, 400)) == (64, 50, 58)


def test_chunk_size_dim3_align():
    # The first dimension is aligned, the second is not
    assert calculate_and_show_chunk_size((121, 87, 65), bytes_per_element=1) == (48, 29, 22)
    assert calculate_and_show_chunk_size((121, 87, 65), bytes_per_element=3) == (48, 29, 22)
    assert calculate_and_show_chunk_size((121, 87, 65), bytes_per_element=2) == (48, 29, 22)
    assert calculate_and_show_chunk_size((121, 87, 65), bytes_per_element=4) == (44, 29, 22)
    assert calculate_and_show_chunk_size((121, 87, 65), bytes_per_element=8) == (42, 29, 22)
    assert calculate_and_show_chunk_size((121, 87, 65), bytes_per_element=16) == (41, 29, 22)


if __name__ == "__main__":
    test_chunk_size_dim2_bounds()
    test_chunk_size_dim2_ratio()
    test_chunk_size_dim2_align()

    test_chunk_size_dim3_align()