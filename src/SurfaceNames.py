#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""Surface name enumerations for biventricular heart models.

This module defines enumerations for surface names used in fiber generation
and Laplace solver configuration.
"""

from enum import Enum


class SurfaceName(Enum):
    """Enumeration of surface names for biventricular heart models.
    
    Attributes:
        EPI: Epicardial surface
        BASE: Base surface (all valves together)
        EPI_APEX: Epicardial apex surface
        ENDO_LV: Left ventricle endocardial surface
        ENDO_RV: Right ventricle endocardial surface
        MV: Mitral valve surface
        AV: Aortic valve surface
        TV: Tricuspid valve surface
        PV: Pulmonary valve surface
    """
    EPI = "epi"
    BASE = "base"
    EPI_APEX = "epi_apex"
    ENDO_LV = "endo_lv"
    ENDO_RV = "endo_rv"
    MV = "mv"
    AV = "av"
    TV = "tv"
    PV = "pv"
    
    @classmethod
    def from_xml_face_name(cls, xml_name):
        """Convert XML face name to SurfaceName enum.
        
        Args:
            xml_name: XML face name (e.g., 'epi_top').
            
        Returns:
            SurfaceName: Corresponding enum value.
        """
        # Map XML face names to enum values
        xml_to_enum = {
            'epi': cls.EPI,
            'epi_top': cls.BASE,
            'epi_apex': cls.EPI_APEX,
            'endo_lv': cls.ENDO_LV,
            'endo_rv': cls.ENDO_RV,
            'mv': cls.MV,
            'av': cls.AV,
            'tv': cls.TV,
            'pv': cls.PV,
        }
        return xml_to_enum.get(xml_name, None)
    
    @classmethod
    def get_required_for_method(cls, method):
        """Get required surface names for a given method.
        
        Args:
            method: Either "bayer" or "doste".
            
        Returns:
            set: Set of required SurfaceName enum values.
        """
        if method == "bayer":
            return {cls.EPI, cls.ENDO_LV, cls.ENDO_RV, cls.BASE, cls.EPI_APEX}
        elif method == "doste":
            return {cls.EPI, cls.ENDO_LV, cls.ENDO_RV, cls.EPI_APEX, 
                   cls.MV, cls.AV, cls.TV, cls.PV}
        else:
            raise ValueError(f"Unknown method: {method}. Use 'bayer' or 'doste'.")
