#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""Test script for the OO implementation of LaplaceSolver and SurfaceProcessor.

This script tests the object-oriented implementation by:
1. Using SurfaceProcessor to generate epi_apex surfaces
2. Using LaplaceSolver with SurfaceName enums
3. Running the Laplace solver on example meshes using both Bayer and Doste methods
4. Comparing the result_001.vtu outputs with reference results in example/.
"""

import os
import argparse
import numpy as np
import pyvista as pv
from src.SurfaceNames import SurfaceName
from src.SurfaceProcessor import SurfaceProcessor
from src.LaplaceSolver import LaplaceSolver


# Fields expected in each method's result_001.vtu
BAYER_FIELDS = ['Trans_EPI', 'Trans_LV', 'Trans_RV', 'Long_AB']
DOSTE_FIELDS = ['Trans_BiV', 'Long_AV', 'Long_MV', 'Long_PV', 'Long_TV', 
                'Weight_LV', 'Weight_RV', 'Trans_EPI', 'Trans_LV', 'Trans_RV']


def compare_meshes(test_mesh, ref_mesh, fields_to_compare, tolerance=1e-6):
    """Compare two meshes and report differences in specified fields.
    
    Args:
        test_mesh: PyVista mesh from the test run.
        ref_mesh: PyVista mesh from the reference output.
        fields_to_compare: List of field names to compare.
        tolerance: Maximum allowed difference (default: 1e-6).
        
    Returns:
        dict: Comparison results for each field.
    """
    results = {}
    
    for field in fields_to_compare:
        if field not in test_mesh.point_data:
            results[field] = {"status": "MISSING", "message": f"Field '{field}' not in test mesh"}
            continue
        if field not in ref_mesh.point_data:
            results[field] = {"status": "MISSING", "message": f"Field '{field}' not in reference mesh"}
            continue
        
        test_data = np.asarray(test_mesh.point_data[field])
        ref_data = np.asarray(ref_mesh.point_data[field])
        
        if test_data.shape != ref_data.shape:
            results[field] = {
                "status": "SHAPE_MISMATCH",
                "message": f"Shape mismatch: test {test_data.shape} vs ref {ref_data.shape}"
            }
            continue
        
        # Compute differences
        diff = np.abs(test_data - ref_data)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        if max_diff <= tolerance:
            results[field] = {
                "status": "PASS",
                "max_diff": max_diff,
                "mean_diff": mean_diff
            }
        else:
            results[field] = {
                "status": "FAIL",
                "max_diff": max_diff,
                "mean_diff": mean_diff,
                "message": f"Max diff {max_diff:.2e} exceeds tolerance {tolerance:.2e}"
            }
    
    return results


def print_comparison_results(results, method_name):
    """Print comparison results in a formatted way."""
    print(f"\n{'='*60}")
    print(f"Comparison Results: {method_name}")
    print('='*60)
    
    all_passed = True
    for field, result in results.items():
        status = result["status"]
        if status == "PASS":
            print(f"  ✓ {field}: PASS (max_diff={result['max_diff']:.2e}, mean_diff={result['mean_diff']:.2e})")
        elif status == "FAIL":
            print(f"  ✗ {field}: FAIL - {result['message']}")
            all_passed = False
        else:
            print(f"  ? {field}: {status} - {result['message']}")
            all_passed = False
    
    print('-'*60)
    if all_passed:
        print(f"  Overall: ALL TESTS PASSED")
    else:
        print(f"  Overall: SOME TESTS FAILED")
    print('='*60)
    
    return all_passed


def test_bayer_method(run_solver=True, generate_apex=True):
    """Test OO implementation with the Bayer method."""
    print("\n" + "="*60)
    print("Testing OO Implementation - BAYER method")
    print("="*60 + "\n")
    
    # Paths - use example for input data
    mesh_path = os.path.abspath("example/truncated/VOLUME.vtu")
    surfaces_dir = os.path.join(os.path.dirname(mesh_path), "mesh-surfaces")
    outdir = os.path.abspath("example/truncated/output_bayer_test")
    ref_dir = os.path.abspath("example/truncated/output_b")
    
    # Surface filenames (not full paths)
    surface_filenames = {
        SurfaceName.EPI: 'EPI.vtp',
        SurfaceName.EPI_APEX: 'EPI_APEX.vtp',
        SurfaceName.BASE: 'BASE.vtp',
        SurfaceName.ENDO_LV: 'LV.vtp',
        SurfaceName.ENDO_RV: 'RV.vtp',
    }
    
    # Generate epi_apex surface if needed
    if generate_apex:
        print("Generating epi_apex surface using SurfaceProcessor...")
        processor = SurfaceProcessor(surfaces_dir, surface_filenames)
        processor.generate_epi_apex()
        print("  ✓ Epi apex surface generated")
    
    # Surface paths (full paths) for LaplaceSolver
    surface_paths = {
        SurfaceName.EPI: os.path.join(surfaces_dir, surface_filenames[SurfaceName.EPI]),
        SurfaceName.EPI_APEX: os.path.join(surfaces_dir, surface_filenames[SurfaceName.EPI_APEX]),
        SurfaceName.BASE: os.path.join(surfaces_dir, surface_filenames[SurfaceName.BASE]),
        SurfaceName.ENDO_LV: os.path.join(surfaces_dir, surface_filenames[SurfaceName.ENDO_LV]),
        SurfaceName.ENDO_RV: os.path.join(surfaces_dir, surface_filenames[SurfaceName.ENDO_RV]),
    }
    
    svfsi_exec = "svmultiphysics "
    
    # Create LaplaceSolver with SurfaceName enums
    print("\nCreating LaplaceSolver with SurfaceName enums...")
    solver = LaplaceSolver(
        mesh_path=mesh_path,
        surface_paths=surface_paths,
        exec_svmultiphysics=svfsi_exec
    )
    
    print(f"Mesh path: {solver.mesh_path}")
    print(f"Output directory: {outdir}")
    print(f"Reference directory: {ref_dir}")
    
    # Run the solver
    if run_solver:
        print("\nRunning Laplace solver...")
        laplace_results_file = solver.run("bayer", outdir)
    else:
        laplace_results_file = os.path.join(outdir, 'result_001.vtu')
        print(f"\nSkipping solver, expecting results at: {laplace_results_file}")
    
    # Compare with reference output
    ref_results_file = os.path.join(ref_dir, "result_001.vtu")
    
    if not os.path.exists(laplace_results_file):
        print(f"\nERROR: Test results not found: {laplace_results_file}")
        return None, False
        
    if not os.path.exists(ref_results_file):
        print(f"\nERROR: Reference results not found: {ref_results_file}")
        return None, False
    
    print(f"\nComparing Laplace solutions:")
    print(f"  Test: {laplace_results_file}")
    print(f"  Ref:  {ref_results_file}")
    
    test_mesh = pv.read(laplace_results_file)
    ref_mesh = pv.read(ref_results_file)
    
    comparison = compare_meshes(test_mesh, ref_mesh, BAYER_FIELDS)
    passed = print_comparison_results(comparison, "Bayer Method - Laplace Solutions")
    
    return test_mesh, passed


def test_doste_method(run_solver=True, generate_apex=True):
    """Test OO implementation with the Doste method."""
    print("\n" + "="*60)
    print("Testing OO Implementation - DOSTE method")
    print("="*60 + "\n")
    
    # Paths - use example for input data
    mesh_path = os.path.abspath("example/ot/mesh-complete.mesh.vtu")
    surfaces_dir = os.path.join(os.path.dirname(mesh_path), "mesh-surfaces")
    outdir = os.path.abspath("example/ot/output_doste_test")
    ref_dir = os.path.abspath("example/ot/output_d")
    
    # Surface filenames (not full paths)
    surface_filenames = {
        SurfaceName.EPI: 'epi.vtp',
        SurfaceName.EPI_APEX: 'epi_apex.vtp',
        SurfaceName.AV: 'av.vtp',
        SurfaceName.MV: 'mv.vtp',
        SurfaceName.TV: 'tv.vtp',
        SurfaceName.PV: 'pv.vtp',
        SurfaceName.BASE: 'top.vtp',
        SurfaceName.ENDO_LV: 'endo_lv.vtp',
        SurfaceName.ENDO_RV: 'endo_rv.vtp',
    }
    
    # Generate epi_apex surface if needed
    if generate_apex:
        print("Generating epi_apex surface using SurfaceProcessor...")
        processor = SurfaceProcessor(surfaces_dir, surface_filenames)
        processor.generate_epi_apex()
        print("  ✓ Epi apex surface generated")
    
    # Surface paths (full paths) for LaplaceSolver
    surface_paths = {
        SurfaceName.EPI: os.path.join(surfaces_dir, surface_filenames[SurfaceName.EPI]),
        SurfaceName.EPI_APEX: os.path.join(surfaces_dir, surface_filenames[SurfaceName.EPI_APEX]),
        SurfaceName.AV: os.path.join(surfaces_dir, surface_filenames[SurfaceName.AV]),
        SurfaceName.MV: os.path.join(surfaces_dir, surface_filenames[SurfaceName.MV]),
        SurfaceName.TV: os.path.join(surfaces_dir, surface_filenames[SurfaceName.TV]),
        SurfaceName.PV: os.path.join(surfaces_dir, surface_filenames[SurfaceName.PV]),
        SurfaceName.BASE: os.path.join(surfaces_dir, surface_filenames[SurfaceName.BASE]),
        SurfaceName.ENDO_LV: os.path.join(surfaces_dir, surface_filenames[SurfaceName.ENDO_LV]),
        SurfaceName.ENDO_RV: os.path.join(surfaces_dir, surface_filenames[SurfaceName.ENDO_RV]),
    }
    
    svfsi_exec = "svmultiphysics "
    
    # Create LaplaceSolver with SurfaceName enums
    print("\nCreating LaplaceSolver with SurfaceName enums...")
    solver = LaplaceSolver(
        mesh_path=mesh_path,
        surface_paths=surface_paths,
        exec_svmultiphysics=svfsi_exec
    )
    
    print(f"Mesh path: {solver.mesh_path}")
    print(f"Output directory: {outdir}")
    print(f"Reference directory: {ref_dir}")
    
    # Run the solver
    if run_solver:
        print("\nRunning Laplace solver...")
        laplace_results_file = solver.run("doste", outdir)
    else:
        laplace_results_file = os.path.join(outdir, 'result_001.vtu')
        print(f"\nSkipping solver, expecting results at: {laplace_results_file}")
    
    # Compare with reference output
    ref_results_file = os.path.join(ref_dir, "result_001.vtu")
    
    if not os.path.exists(laplace_results_file):
        print(f"\nERROR: Test results not found: {laplace_results_file}")
        return None, False
        
    if not os.path.exists(ref_results_file):
        print(f"\nERROR: Reference results not found: {ref_results_file}")
        return None, False
    
    print(f"\nComparing Laplace solutions:")
    print(f"  Test: {laplace_results_file}")
    print(f"  Ref:  {ref_results_file}")
    
    test_mesh = pv.read(laplace_results_file)
    ref_mesh = pv.read(ref_results_file)
    
    comparison = compare_meshes(test_mesh, ref_mesh, DOSTE_FIELDS)
    passed = print_comparison_results(comparison, "Doste Method - Laplace Solutions")
    
    return test_mesh, passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test OO implementation (LaplaceSolver and SurfaceProcessor)")
    parser.add_argument(
        "--method", 
        choices=["bayer", "doste", "both"],
        default="both",
        help="Which method to test (default: both)"
    )
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="Skip running the solver, compare existing results"
    )
    parser.add_argument(
        "--no-apex",
        action="store_true",
        help="Skip generating epi_apex surface (assume it already exists)"
    )
    args = parser.parse_args()
    
    run_solver = not args.no_run
    generate_apex = not args.no_apex
    results = {}
    
    if args.method in ["bayer", "both"]:
        _, passed = test_bayer_method(run_solver, generate_apex)
        results["bayer"] = passed
        
    if args.method in ["doste", "both"]:
        _, passed = test_doste_method(run_solver, generate_apex)
        results["doste"] = passed
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    all_passed = True
    for method, passed in results.items():
        if passed is None:
            print(f"  {method.upper()}: ERROR - Could not compare")
            all_passed = False
        elif passed:
            print(f"  {method.upper()}: ✓ PASSED")
        else:
            print(f"  {method.upper()}: ✗ FAILED")
            all_passed = False
    
    print("="*60)
    if all_passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
    print("="*60)
