
"""Defines the interpolation functions to fill the missing landmarks in the landmarks array"""

def linear_interpolation(start_idx: int, end_idx: int, landmarks):
    """Defines a linear interpolation function to interpolate missing landmarks between indices

    Args:
        start_idx (int): An integer defining the starting index
        end_idx (int): An integer defining the stopping index
        landmarks: An array of size 68 containing the (x,y) values of the facial landmarks

    Returns:
        landmarks array after the missing points have been interpolated.
    """
    start_landmarks = landmarks[start_idx]
    end_landmarks = landmarks[end_idx]
    delta_idx = end_idx - start_idx
    delta_landmarks = end_landmarks - start_landmarks

    # Apply linear interpolation formula
    for idx in range(1, delta_idx):
        landmarks[idx + start_idx] = start_landmarks + delta_landmarks/delta_idx * idx
    return landmarks


def landmarks_interpolation(landmarks):
    """Adds the missing landmarks to the landmarks array

    Args:
        landmarks: An array containing all the detected landmarks

    Returns:
        landmarks array filled in with missing landmarks
    """
    # Obtain indices of all the valid landmarks (i.e landmarks not None)
    valid_landmarks_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    
    if not valid_landmarks_idx:
        return

    # For middle parts of the landmarks array
    for idx in range(1, len(valid_landmarks_idx)):
        # If the valid landmarks indices are adjacent then skip to next iteration
        if valid_landmarks_idx[idx]-valid_landmarks_idx[idx-1] > 1:
            landmarks = linear_interpolation(start_idx=valid_landmarks_idx[idx-1],
                                            end_idx=valid_landmarks_idx[idx],
                                            landmarks=landmarks)

    # For beginning and ending parts of the landmarks array
    valid_landmarks_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    landmarks[:valid_landmarks_idx[0]] = [landmarks[valid_landmarks_idx[0]]] * valid_landmarks_idx[0]
    landmarks[valid_landmarks_idx[-1]:] = [landmarks[valid_landmarks_idx[-1]]] * (len(landmarks) - valid_landmarks_idx[-1])

    return landmarks
