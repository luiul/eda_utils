def test_wavg_grouped():
    # Test 1: Basic functionality
    df = pd.DataFrame({
        'group': ['A', 'A', 'A', 'B', 'B', 'B'],
        'values': [10, 20, 30, 40, 50, 60],
        'weights': [1, 2, 3, 1, 2, 3]
    })
    result = wavg_grouped(df, 'values', 'weights', 'group')
    expected_result = pd.DataFrame({
        'group': ['A', 'B'],
        'wavg': [23.33, 53.33]
    })
    pd.testing.assert_frame_equal(result.round(2), expected_result, check_dtype=False)

    # Test 2: NaN values in the 'values' and 'weights' columns
    df_nan = pd.DataFrame({
        'group': ['A', 'A', 'A', 'B', 'B', 'B'],
        'values': [10, 20, np.nan, 40, 50, 60],
        'weights': [1, 2, np.nan, 1, 2, 3]
    })
    expected_result = pd.DataFrame({
        'group': ['A', 'B'],
        'wavg': [16.67, 53.33]
    })
    result_nan = wavg_grouped(df_nan, 'values', 'weights', 'group')
    pd.testing.assert_frame_equal(result_nan.round(2), expected_result, check_dtype=False)

    # Test 3: 'weights' sum to zero for a group
    df_zero_weights = pd.DataFrame({
        'group': ['A', 'A', 'A', 'B', 'B', 'B'],
        'values': [10, 20, 30, 40, 50, 60],
        'weights': [1, 2, 0, 0, 0, 0]
    })
    try:
        wavg_grouped(df_zero_weights, 'values', 'weights', 'group')
    except ValueError as e:
        assert str(e) == "One of the groups has a sum of weights equal to zero, cannot perform division by zero.", "Unexpected error message."

    # Test 4: Multi-index grouping
    df_multi = pd.DataFrame({
        'group1': ['A', 'A', 'B', 'B', 'C', 'C'],
        'group2': ['X', 'Y', 'X', 'Y', 'X', 'Y'],
        'values': [10, 20, 30, 40, 50, 60],
        'weights': [1, 2, 3, 1, 2, 3]
    })
    result_multi = wavg_grouped(df_multi, 'values', 'weights', ['group1', 'group2'])
    expected_multi = pd.DataFrame({
        'group1': ['A', 'A', 'B', 'B', 'C', 'C'],
        'group2': ['X', 'Y', 'X', 'Y', 'X', 'Y'],
        'wavg': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    })
    pd.testing.assert_frame_equal(result_multi, expected_multi, check_dtype=False)

test_wavg_grouped()
