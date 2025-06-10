"""Basic tests for AIMS."""


def calculate_average(numbers):
    """Calculate the average of a list of numbers."""
    return sum(numbers) / len(numbers)


def test_calculate_average():
    """Test the calculate_average function."""
    # Test with a list of integers
    numbers = [1, 2, 3, 4, 5]
    result = calculate_average(numbers)
    assert result == 3.0

    # Test with a list of floats
    numbers = [1.5, 2.5, 3.5]
    result = calculate_average(numbers)
    assert result == 2.5


def test_simple_math():
    """Test simple math operations."""
    assert 1 + 1 == 2
    assert 5 - 3 == 2
    assert 2 * 3 == 6
    assert 6 / 3 == 2
    assert 2**3 == 8
