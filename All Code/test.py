import cv2
import numpy as np
import os
from pathlib import Path

def test_new_conservative_methods(image_path):
    """Test the ultra-conservative methods specifically."""
    from ThresholdManager import ThresholdManager
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    tm = ThresholdManager()
    
    output_dir = Path("./threshold_debug")
    
    methods = {
        'combo30_nuskhuri_conservative': tm.combo30_nuskhuri_conservative,
        'self.combo31_ultra_conservative': tm.combo31_ultra_conservative,
        'self.ultra': tm.ultra,
        'self.ultra_enhanced': tm.ultra_enhanced
    }
    
    print("\n" + "="*70)
    print("TESTING ULTRA-CONSERVATIVE METHODS")
    print("="*70)
    
    for name, method in methods.items():
        binary = method(img)
        white_pct = np.mean(binary == 255)
        nb = cv2.connectedComponentsWithStats(binary, connectivity=8)[0] - 1
        
        print(f"\n{name}:")
        print(f"  White pixels: {white_pct:.1%}")
        print(f"  Components: {nb}")
        print(f"  Target: 5-15% white, 100-500 components")
        
        cv2.imwrite(str(output_dir / f"{name}.png"), binary)
        
        # Also save with boxes
        _, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for i in range(1, min(nb+1, 1000)):  # limit to first 1000 for visibility
            x, y, w, h = stats[i][:4]
            cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 1)
        
        cv2.imwrite(str(output_dir / f"{name}_with_boxes.png"), vis)
    
    print(f"\nResults saved to {output_dir}/")

def analyze_document_quality(image_path):
    """
    Determines if a document is fundamentally parseable
    and identifies specific problems.
    """
    print("="*70)
    print("MANUSCRIPT QUALITY ANALYSIS")
    print("="*70)
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"ERROR: Cannot read {image_path}")
        return None
    
    h, w = img.shape
    print(f"\n1. IMAGE PROPERTIES")
    print(f"   Dimensions: {w}x{h} pixels")
    print(f"   File: {os.path.basename(image_path)}")
    
    # Analyze histogram
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist = hist.flatten() / hist.sum()
    
    print(f"\n2. INTENSITY DISTRIBUTION")
    print(f"   Min: {img.min()}, Max: {img.max()}, Mean: {img.mean():.1f}")
    
    # Check if image is too uniform (low contrast)
    std = img.std()
    print(f"   Std Dev: {std:.1f}")
    if std < 20:
        print("   ⚠️  PROBLEM: Very low contrast - image is nearly uniform")
        print("      → Document may be too faded or scan quality too poor")
    
    # Check dynamic range
    p1, p99 = np.percentile(img, [1, 99])
    dynamic_range = p99 - p1
    print(f"   Dynamic range (1-99 percentile): {dynamic_range:.1f}")
    if dynamic_range < 50:
        print("   ⚠️  PROBLEM: Narrow dynamic range")
        print("      → Very little separation between ink and background")
    
    # Check for bimodal distribution (good for thresholding)
    print(f"\n3. HISTOGRAM ANALYSIS")
    peaks = []
    for i in range(10, 246):
        if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > 0.002:
            peaks.append((i, hist[i]))
    
    print(f"   Found {len(peaks)} significant peaks")
    if len(peaks) < 2:
        print("   ⚠️  PROBLEM: Non-bimodal histogram")
        print("      → No clear separation between foreground/background")
        print("      → Otsu and similar methods will fail")
    else:
        print(f"   ✓ Peaks at intensities: {[p[0] for p in peaks]}")
    
    # Estimate ink percentage
    print(f"\n4. INK ESTIMATION (multiple thresholds)")
    for thresh_val in [100, 127, 150, 180]:
        _, binary = cv2.threshold(img, thresh_val, 255, cv2.THRESH_BINARY_INV)
        ink_ratio = np.mean(binary == 255)
        print(f"   Threshold {thresh_val}: {ink_ratio:.2%} ink")
        
    # Try Otsu
    otsu_thresh, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    print(f"   Otsu's threshold: {otsu_thresh}")
    
    # Check for illumination issues
    print(f"\n5. ILLUMINATION UNIFORMITY")
    # Divide into 4x4 grid, check mean of each cell
    cell_h, cell_w = h // 4, w // 4
    cell_means = []
    for i in range(4):
        for j in range(4):
            cell = img[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            cell_means.append(cell.mean())
    
    illumination_var = np.std(cell_means)
    print(f"   Std dev across grid: {illumination_var:.1f}")
    if illumination_var > 20:
        print("   ⚠️  PROBLEM: Uneven illumination")
        print(f"      → Brightest region: {max(cell_means):.1f}")
        print(f"      → Darkest region: {min(cell_means):.1f}")
        print("      → Need illumination correction preprocessing")
    else:
        print("   ✓ Illumination fairly uniform")
    
    # Edge detection check (are there detectable features?)
    print(f"\n6. FEATURE DETECTABILITY")
    edges = cv2.Canny(img, 50, 150)
    edge_density = np.mean(edges > 0)
    print(f"   Edge pixel density: {edge_density:.2%}")
    if edge_density < 0.01:
        print("   ⚠️  PROBLEM: Very few edges detected")
        print("      → Image may be too blurred or low contrast")
    elif edge_density > 0.3:
        print("   ⚠️  PROBLEM: Too many edges detected")
        print("      → Likely very noisy image or texture artifacts")
    else:
        print("   ✓ Reasonable edge structure present")
    
    # Overall assessment
    print(f"\n{'='*70}")
    print("OVERALL ASSESSMENT")
    print("="*70)
    
    problems = []
    if std < 20:
        problems.append("low_contrast")
    if dynamic_range < 50:
        problems.append("narrow_range")
    if len(peaks) < 2:
        problems.append("non_bimodal")
    if illumination_var > 20:
        problems.append("uneven_illumination")
    if edge_density < 0.01:
        problems.append("no_features")
    
    if not problems:
        print("✓ Document appears parseable")
        print("  → Issue is likely in threshold/segmentation parameters")
    else:
        print("⚠️  Document has fundamental quality issues:")
        for p in problems:
            print(f"   - {p}")
        print("\n  These issues may make traditional OCR difficult.")
    
    return {
        'parseable': len(problems) == 0,
        'problems': problems,
        'std': std,
        'dynamic_range': dynamic_range,
        'illumination_var': illumination_var,
        'edge_density': edge_density
    }


def test_thresholds_systematically(image_path):
    """
    Test different threshold approaches and show actual results.
    """
    print("\n" + "="*70)
    print("THRESHOLD METHOD COMPARISON")
    print("="*70)
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    output_dir = Path("./threshold_debug")
    output_dir.mkdir(exist_ok=True)
    
    # Test methods
    methods = {}
    
    # 1. Simple Otsu
    _, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    methods['otsu'] = otsu
    
    # 2. Adaptive Mean (multiple window sizes)
    for block_size in [11, 21, 41, 71]:
        adapt = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
            cv2.THRESH_BINARY_INV, block_size, 5
        )
        methods[f'adaptive_mean_{block_size}'] = adapt
    
    # 3. Adaptive Gaussian
    for block_size in [11, 21, 41, 71]:
        adapt = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, block_size, 5
        )
        methods[f'adaptive_gauss_{block_size}'] = adapt
    
    # 4. With preprocessing
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(img)
    _, otsu_clahe = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    methods['otsu_with_clahe'] = otsu_clahe
    
    # Analyze each
    print(f"\n{'Method':<25} {'White%':<10} {'Components':<12} {'Status'}")
    print("-"*70)
    
    results = []
    for name, binary in methods.items():
        white_pct = np.mean(binary == 255)
        nb = cv2.connectedComponentsWithStats(binary, connectivity=8)[0]
        
        status = ""
        if white_pct > 0.5:
            status = "⚠️  inverted?"
        elif white_pct < 0.05:
            status = "⚠️  too sparse"
        elif white_pct > 0.4:
            status = "⚠️  too dense"
        elif nb > 1000:
            status = "⚠️  noisy"
        elif nb < 20:
            status = "⚠️  empty"
        else:
            status = "✓ reasonable"
        
        print(f"{name:<25} {white_pct:>6.1%}    {nb:>8}      {status}")
        
        # Save
        cv2.imwrite(str(output_dir / f"{name}.png"), binary)
        
        results.append((name, white_pct, nb, status))
    
    # Recommendation
    print(f"\n{'='*70}")
    print("RECOMMENDATION")
    print("="*70)
    
    # Find methods with reasonable metrics
    good = [(n, w, c) for n, w, c, s in results 
            if 0.05 < w < 0.4 and 20 < c < 1000]
    
    if good:
        best = min(good, key=lambda x: abs(x[1] - 0.15))  # prefer ~15% ink
        print(f"\nBest method: {best[0]}")
        print(f"  - {best[1]:.1%} ink coverage")
        print(f"  - {best[2]} components")
        print(f"\nCheck: {output_dir}/{best[0]}.png")
    else:
        print("\n⚠️  NO methods produced reasonable results")
        print("This suggests:")
        print("  1. Document quality is too poor for traditional methods")
        print("  2. OR parameters need extreme tuning")
        print("  3. OR image needs manual preprocessing")
        
        print(f"\nAll outputs saved to: {output_dir}/")
        print("Manually inspect them to understand what's happening.")


def visualize_problem_areas(image_path):
    """
    Create visual diagnosis of where processing fails.
    """
    print("\n" + "="*70)
    print("CREATING VISUAL DIAGNOSTICS")
    print("="*70)
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    output_dir = Path("./threshold_debug")
    
    # Create comparison view
    fig_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Try one threshold method
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    # Find components
    nb, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    # Color-code by size
    colored = np.zeros_like(fig_img)
    for i in range(1, nb):
        area = stats[i, cv2.CC_STAT_AREA]
        mask = (labels == i)
        
        if area < 20:
            colored[mask] = [0, 0, 255]  # tiny = red (noise)
        elif area < 100:
            colored[mask] = [0, 255, 255]  # small = yellow (maybe letters)
        elif area < 1000:
            colored[mask] = [0, 255, 0]  # medium = green (likely letters)
        else:
            colored[mask] = [255, 0, 0]  # large = blue (artifacts)
    
    # Save
    cv2.imwrite(str(output_dir / "component_size_analysis.png"), colored)
    
    print(f"\n✓ Saved component size visualization")
    print(f"  Red = noise (<20px)")
    print(f"  Yellow = small (20-100px)")  
    print(f"  Green = medium (100-1000px) ← should be letters")
    print(f"  Blue = large (>1000px) ← artifacts")
    print(f"\nFile: {output_dir}/component_size_analysis.png")


if __name__ == "__main__":
    # CHANGE THIS to your image path
    test_image = "/Users/sunnysideup/Desktop/untitled folder/page_11_nuskhuri.png"
    
    if not os.path.exists(test_image):
        print(f"ERROR: File not found: {test_image}")
        print("Update the test_image path in this script.")
    else:
        # Run full diagnostics
        quality = analyze_document_quality(test_image)
        test_new_conservative_methods(test_image)
        visualize_problem_areas(test_image)
        
        print("\n" + "="*70)
        print("NEXT STEPS")
        print("="*70)
        print("\n1. Check ./threshold_debug/ for all output images")
        print("2. Find which threshold image looks cleanest")
        print("3. Report back with:")
        print("   - Which method looked best")
        print("   - What the component_size_analysis shows")
        print("   - Any specific patterns you see")