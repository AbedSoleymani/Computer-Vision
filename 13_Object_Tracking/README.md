# Object Tracking
Object tracking enables machines to follow and understand the motion of objects in video streams. This repository focuses on a state-of-the-art object tracking method, namely DeepSORT, which combines traditional tracking techniques with deep learning for improved accuracy and speed.

## Historical Context
Mean-shift and optical flow are traditional object tracking methods. However, they face challenges such as high computational complexity, sensitivity to noise, and tracking loss during occlusion. These limitations led to the evolution of more advanced tracking algorithms.

Kalman Filter in SORT
Kalman Filter plays a key role in the SORT (Simple Online and Realtime Tracking) algorithm. In the estimation stage, the Kalman filter employs a linear velocity model with a state vector that includes components like position, scale, and aspect ratio. This helps to predict the object's state at each frame.

## SORT: Four Stages
1. Detection: In the detection stage, the algorithm receives object bounding boxes from a detection module. To refine these detections, the Kalman filter is employed. The state vector, $x$, includes various elements representing the object's characteristics. Mathematically, the update process can be expressed as:

$$
x_k = F \cdot x_{k-1} + B \cdot u_k + w_k
$$

Where:
- $x_k$ is the state vector at time $k$,
- $F$ is the state transition matrix,
- $B$ is the control-input matrix,
- $u_k$ is the control vector (which could represent the acceleration),
- $w_k$ is the process noise.

The state vector ($x$) typically includes components such as:
- $u$ and $v$: Horizontal and vertical pixel locations of the target center.
- $s$: Scale - area of the target's bounding box.
- $r$: Aspect ratio - the ratio of width to height.
- $\dot{u}$, $\dot{v}$, $\dot{s}$: Velocities corresponding to $u$, $v$, and $s$.

2. Estimation: The Kalman filter aids in estimating the state of the tracked object. Given the observed measurements, the Kalman gain ($K_k$) is calculated to weigh the difference between the predicted state ($\hat{x}_k^-$) and the measurement ($z_k$). Mathematically, the estimation is represented as:

$$
\hat{x}_k = \hat{x}_k^- + K_k \cdot (z_k - H \cdot \hat{x}_k^-)
$$

Where:
- $\hat{x}_k^-$ is the predicted state,
- $H$ is the observation matrix.

3. Association: Target association involves linking the current frame's detections to existing tracks. This is achieved by computing the assignment cost matrix using the Intersection over Union (IoU) between bounding boxes. The Hungarian algorithm optimally matches detections to tracks, minimizing the total cost.

**Hungarian Algorithm:**
The Hungarian algorithm solves the assignment problem by finding the optimal assignment that minimizes the total cost. Given the cost matrix (association cost between tracks and detections), the algorithm efficiently determines the optimal assignment, ensuring each track is linked to the most likely detection. It is noteworthy that the time complexity of this algorithm is $O(n^3)$, which makes it much more efficient than the brute-force method with a time complexity of $O(n!)$. To perform this algorithm in Python, we can simply use Pscipy library as follows:

```python
import numpy as np
from scipy.optimize import linear_sum_assignment

def hungarian_algorithm(cost_matrix):
    """
    Apply the Hungarian Algorithm to solve the assignment problem.

    Parameters:
    - cost_matrix: 2D array representing the cost of assigning each agent to each task.
      rows: tracks (i.e., agent)
      columns: detections (i.e., cost of the task)

    Returns:
    - row_ind, col_ind: Arrays of row and column indices representing the optimal assignment.
    """
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return row_ind, col_ind

cost_matrix = np.array([[10, 20, 15], [15, 25, 30], [25, 30, 35]])

row_indices, col_indices = hungarian_algorithm(cost_matrix)

print("Optimal Assignment:")
for i in range(len(row_indices)):
    print(f"Agent {row_indices[i]} is assigned to Task {col_indices[i]} with cost {cost_matrix[row_indices[i], col_indices[i]]}")
```

which returns:

```bash
Optimal Assignment:
Agent 0 is assigned to Task 2 with cost 15
Agent 1 is assigned to Task 0 with cost 15
Agent 2 is assigned to Task 1 with cost 30
```

4. Track Identity Lifecycle: The track identity lifecycle involves track initialization, handling uncertainties, and managing lost frames. The state vector elements, such as velocities, provide information crucial for predicting the object's future state. Tracks undergo a probationary period to accumulate evidence. Tracks are terminated if they are not detected for a certain number of frames (Threshold for Track Lost - TLost). Reappearing objects resume tracking under new identities.

These stages collectively form the SORT algorithm, providing a comprehensive framework for online and real-time object tracking.

## DeepSORT Integration
To enhance the tracking capabilities of SORT (Simple Online and Realtime Tracking), DeepSORT introduces a novel approach by incorporating information about the appearance of tracked objects. This integration addresses some of the limitations faced by traditional tracking methods, such as occlusion and inaccurate association during object detection.

**Appearance Descriptor and Feature Vector:**
DeepSORT leverages a dense layer to generate an appearance descriptor, often referred to as a feature vector, for each tracked object. This descriptor captures unique characteristics of the object's appearance, creating a compact representation of its visual features. The dense layer processes the raw input, extracting relevant information and producing a feature vector that encodes the object's appearance.

**Nearest Neighbor Queries:**
The obtained appearance descriptors are then used to perform nearest neighbor queries. These queries involve searching for objects with similar visual characteristics within the feature space. DeepSORT establishes a measurement-to-track association by identifying the nearest neighbors in appearance, facilitating robust and accurate tracking across frames.

**Improving Tracking Robustness:**
By incorporating appearance-based information, DeepSORT significantly improves tracking robustness. The system becomes more resilient to challenges such as occlusion, lighting variations, and changes in object orientation. This additional feature helps maintain the identity of objects even when traditional tracking methods might struggle.

Below is the demonstration of the object tracking performance using YOLOv8 and DeepSORT:

![output](https://github.com/AbedSoleymani/Computer-Vision/assets/72225265/84c6158c-7845-422d-b80f-eb7a35adc117)
