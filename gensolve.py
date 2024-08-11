# Import required libraries
import numpy as np
import pandas as pd
import cv2
from scipy.interpolate import UnivariateSpline, interp1d
import matplotlib.pyplot as plt
import svgwrite
import zipfile
from io import BytesIO

# Unzip the dataset
with zipfile.ZipFile("problems.zip","r") as zip_ref:
    zip_ref.extractall() # problems folder will be created

# Utility functions

def read_csv(csv_path):
    """
    CSV Data Loading and Processing
    """
    np_path_XYs = np.genfromtxt(csv_path ,delimiter = ',')
    path_XYs = []
    for i in np.unique(np_path_XYs [: , 0]):
        npXYs = np_path_XYs[np_path_XYs[: , 0] == i ][: , 1:]
        XYs = []
        for j in np.unique(npXYs[: , 0]):
            XY = npXYs[npXYs[: , 0] == j ][: , 1:]
            XYs.append(XY)
        path_XYs.append(XYs)
    return path_XYs

def plot(paths_XYs):
    """Plotting the Curves"""
    fig , ax = plt.subplots(tight_layout = True ,figsize =(8 , 8))
    for i , XYs in enumerate(paths_XYs):
        # c = colours [ i % len( colours )]
        for XY in XYs:
            ax.plot(XY[: , 0] ,XY[: , 1] ,linewidth =2)
    ax.set_aspect("equal")
    plt.show()


def polylines2svg(paths_XYs, svg_path):
    """Converting Polylines to SVG"""
    W, H = 0, 0
    for path_XYs in paths_XYs:
        for XY in path_XYs:
            W, H = max(W, np.max(XY[:, 0])), max(H, np.max(XY[:, 1]))
    padding = 0.1
    W, H = int(W + padding * W), int(H + padding * H)

    dwg = svgwrite.Drawing(svg_path, profile="tiny", shape_rendering="crispEdges")
    group = dwg.g()
    colors = ["red", "blue", "green", "yellow", "purple", "orange", "cyan", "magenta"]

    for i, path in enumerate(paths_XYs):
        path_data = []
        for XY in path:
            path_data.append("M {} {}".format(XY[0, 0], XY[0, 1]))
            for j in range(1, len(XY)):
                path_data.append("L {} {}".format(XY[j, 0], XY[j, 1]))
            if not np.allclose(XY[0], XY[-1]):
                path_data.append("Z")
        c = colors[i % len(colors)]
        group.add(dwg.path(d=" ".join(path_data), fill=c, stroke="none", stroke_width=2))

    dwg.add(group)
    dwg.save()

    png_path = svg_path.replace('.svg', '.png')
    fact = 1
    if min(H, W) != 0:
        fact = max(1, 1024 // min(H, W))


def smooth_points(x, y, s=0):
    """Smoothing function"""
    spline_x = UnivariateSpline(range(len(x)), x, s=s)
    spline_y = UnivariateSpline(range(len(y)), y, s=s)
    return spline_x(range(len(x))), spline_y(range(len(y)))

def interpolate_points(x, y, num_points):
    """Interpolation function"""
    t = np.linspace(0, 1, len(x))
    f_x = interp1d(t, x, kind="linear")
    f_y = interp1d(t, y, kind="linear")
    t_new = np.linspace(0, 1, num_points)
    return f_x(t_new), f_y(t_new)


def points_to_image(points, width=1000, height=1000):
    """Converting points to image"""
    img = np.zeros((height, width), dtype=np.uint8)
    for x, y in points:
        if 0 <= int(y) < height and 0 <= int(x) < width:
            img[int(y), int(x)] = 255
    return img


def detect_shapes(img):
    """Detecting shapes in an image"""
    shapes = []
    edges = cv2.Canny(img.copy(), 0, 50)
    edges_line = cv2.GaussianBlur(edges.copy(), (15, 15), 0)

    # Detect lines using Probabilistic Hough Line Transform
    lines = cv2.HoughLinesP(edges_line, 1, np.pi / 2, threshold=200, minLineLength=0, maxLineGap=100)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                shapes.append(("Line", np.array([[x1, y1], [x2, y2]])))

    edges = img.copy()
    # Find contours
    contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < 500:  # Adjust the threshold as needed
            continue

        epsilon = 0.03 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # If shape is a triangle
        if len(approx) == 3:
            shapes.append(("Triangle", approx))
        elif len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            shape = "Square" if 0.85 <= aspect_ratio <= 1.15 else "Rectangle"
            shapes.append((shape, approx))
        elif len(approx) > 4:
            area = cv2.contourArea(contour)
            (x, y), radius = cv2.minEnclosingCircle(contour)
            circularity = area / (np.pi * radius * radius)
            if 0.75 <= circularity <= 1.25:
                center = (int(x), int(y))
                radius = int(radius)
                shapes.append(("Circle", (center, radius)))
            else:
                shapes.append(("Polygon", approx))

            if len(approx) >= 6:
                ellipse = cv2.fitEllipse(contour)
                center, axes, angle = ellipse
                axes = (int(axes[0] / 2), int(axes[1] / 2))
                ellipse_contour = cv2.ellipse2Poly(
                    center=(int(center[0]), int(center[1])),
                    axes=axes,
                    angle=int(angle),
                    arcStart=0,
                    arcEnd=360,
                    delta=2
                )
                ellipse_contour = np.array(ellipse_contour)
                distance = cv2.pointPolygonTest(ellipse_contour, center, True)
                if abs(distance) < 40:
                    shapes.append(("Ellipse", ellipse_contour))
            # If shape is star
            if len(approx) >= 10:
                shapes.append(("Star", approx))

    # Select the shape with the highest probability
    shape_priorities = {
        "Circle": 1,
        "Square": 2,
        "Rectangle": 3,
        "Triangle": 4,
        "Star": 5,
        "Polygon": 6,
        "Ellipse": 7,
        "Line": 8,
    }

    if shapes:
        shapes = sorted(shapes, key=lambda s: shape_priorities.get(s[0], 9))
        most_probable_shape = shapes[0]
        return [most_probable_shape]

    return shapes

def draw_shapes(img, shapes, curve_points=None):
    if len(img.shape) == 2:
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_color = img.copy()

    blank_image = np.zeros_like(img_color)

    if shapes:
        for shape, contour in shapes:
            color = (255, 255, 255)  # White

            if shape == "Circle":
                center, radius = contour
                cv2.circle(blank_image, center, radius, color, 1)
            else:
                cv2.drawContours(blank_image, [contour], -1, color, 1)
    else:
        if curve_points is not None:
            color = (255, 255, 255)  # White
            cv2.polylines(blank_image, [curve_points], isClosed=False, color=color, thickness=1)

    return blank_image


def combine_images(images, positions, width=1000, height=1000):
    combined_image = np.zeros((height, width, 3), dtype=np.uint8)
    for img, (x, y) in zip(images, positions):
        h, w = img.shape[:2]
        x = max(0, min(x, width - w))
        y = max(0, min(y, height - h))
        mask = img != 0
        combined_image[y : y + h, x : x + w][mask] = img[mask]
    return combined_image


#################### TASK 2 ##############################################

def draw_symmetry_lines(image, contour,lines_count = 2):
    """Draw 4 symmetry lines for circles and 6 for squares and stars."""
    M = cv2.moments(contour)
    if M['m00'] != 0:
        # Calculate centroid of the contour
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        # Determine shape based on circularity
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)
        circularity = 4 * np.pi * area / (perimeter ** 2)

        if 0.8 < circularity <= 1.2:
            # Circle: Draw 4 lines
            num_lines = 4
        else:
            # Squares and Stars: Draw 6 lines
            num_lines = 6

        # Draw symmetry lines
        for i in range(num_lines):
            theta = i * 2 * np.pi / num_lines  # Evenly space the lines
            length = 75  # Adjust length of the lines
            x_end = int(cx + length * np.cos(theta))
            y_end = int(cy + length * np.sin(theta))
            x_start = int(cx - length * np.cos(theta))
            y_start = int(cy - length * np.sin(theta))
            cv2.line(image, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)


def detect_shapes_and_draw_correct_lines(image):
    """Detect shapes and draw corrected symmetry lines."""
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert to BGR for color drawing

    for contour in contours:
        # Draw the contour
        cv2.drawContours(output, [contour], -1, (0, 255, 0), 2)

        # Draw the correct symmetry lines
        draw_symmetry_lines(output, contour)

    return output

############################## TASK 2 END ##########################

import streamlit as st

# Streamlit App
st.title("Curve Regularization and Beautification")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Curve Processing", "Shape Symmetry"])


if page == "Curve Processing":
    st.header("Curve Processing")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:

    # Process each curve
    # filename = input("Enter filename along without svg extension")

        # Process the uploaded file
        df = pd.read_csv(uploaded_file, header=None, names=["Curve", "Shape", "X", "Y"])
        
        curves = df.groupby(["Curve", "Shape"])
        images = []
        positions = []

        for curve_id, group in curves:

            x, y = group['X'].values, group['Y'].values
            x_smooth, y_smooth = smooth_points(x, y, s=0)
            x_interp, y_interp = interpolate_points(x_smooth, y_smooth, num_points=1000)


            points = np.vstack((x_interp, y_interp)).T
            positions.append((int(x.min()), int(y.min())))  
            # Store original positions for combining

            img = points_to_image(points)
            # cv2.imwrite(f"without_shapes_detected_{curve_id}.png", img)
            shapes = detect_shapes(img)

            # ########################### Plotting ####################################
            # # Create a blank image with white background
            # img_array = np.ones((1000, 1000), dtype=np.uint8)

            # # Set the pixels corresponding to the coordinates to black
            # for xi, yi in zip(x, y):
            #     if 0 <= int(yi) < 1000 and 0 <= int(xi) < 1000:
            #         img_array[int(yi), int(xi)] = 255
            # # Plot the image
            # plt.imshow(img_array, cmap="gray", origin="upper")
            # plt.title(f"Shape: {shapes[0][0]}")
            # plt.show()

            # ######################################################################

            # If no shapes are detected, use the original curve points
            img_with_shapes = draw_shapes(img, shapes, curve_points=np.int32(points))
            images.append(img_with_shapes)

        # Combine all images into one large image
        combined_image = combine_images(images, positions, width=1000, height=1000)
        # cv2.imwrite(f"solutions/{filename}_sol.png", combined_image)

        # Display the image in Streamlit
        st.image(combined_image, caption="Processed Image", use_column_width=True)

        # Download the image
        is_success, buffer = cv2.imencode(".png", combined_image)
        if is_success:
            st.download_button(
                label="Download Image",
                data=BytesIO(buffer),
                file_name="processed_image.png",
                mime="image/png"
            )

elif page == "Shape Symmetry":
    st.header("Shape Symmetry Correction")
    uploaded_image = st.file_uploader("Upload an image (PNG or SVG)", type=["png", "svg"])
    if uploaded_image is not None:
        # Handle SVG files
        if uploaded_image.name.endswith(".svg"):
            import cairosvg

            svg_data = uploaded_image.read()
            filename = uploaded_image.name.split(".")[0]
            temp_png = f"temp_{filename}.png"
            cairosvg.svg2png(bytestring=svg_data, write_to=temp_png)
            image = cv2.imread(temp_png, cv2.IMREAD_GRAYSCALE)
        else:
            # Directly read PNG files
            image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        
        corrected_image = detect_shapes_and_draw_correct_lines(image)

        # Display the image in Streamlit
        st.image(cv2.cvtColor(corrected_image, cv2.COLOR_BGR2RGB), caption="Corrected Image", use_column_width=True)
        
        # Download the corrected image
        is_success, buffer = cv2.imencode(".png", corrected_image)
        if is_success:
            st.download_button(
                label="Download Corrected Image",
                data=BytesIO(buffer),
                file_name="corrected_image.png",
                mime="image/png"
            )