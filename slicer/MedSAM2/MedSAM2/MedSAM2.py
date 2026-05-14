import logging
import os
import json
from typing import Annotated, Optional

import vtk

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)

from slicer import vtkMRMLScalarVolumeNode

import numpy as np
import tempfile
import threading
import requests
import time


#
# MedSAM2
#


class MedSAM2(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("MedSAM2") 
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Segmentation")]
        self.parent.dependencies = []  
        self.parent.contributors = ["Reza Asakereh (University Health Network)", "Sumin Kim (University of Toronto)", "Jun Ma (University Health Network)"]  
        self.parent.helpText = _("""
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#MedSAM2">module documentation</a>.
""")
        self.parent.acknowledgementText = _("""
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
""")



#
# MedSAM2ParameterNode
#


@parameterNodeWrapper
class MedSAM2ParameterNode:
    inputVolume: vtkMRMLScalarVolumeNode
    imageThreshold: Annotated[float, WithinRange(-100, 500)] = 100
    invertThreshold: bool = False
    thresholdedVolume: vtkMRMLScalarVolumeNode
    invertedVolume: vtkMRMLScalarVolumeNode


#
# MedSAM2Widget
#


class MedSAM2Widget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    def __init__(self, parent=None) -> None:
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None

    def setup(self) -> None:
        ScriptedLoadableModuleWidget.setup(self)

        uiWidget = slicer.util.loadUI(self.resourcePath("UI/MedSAM2.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        uiWidget.setMRMLScene(slicer.mrmlScene)

        self.logic = MedSAM2Logic()
        self.logic.widget = self

        # Connections
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Preprocessing
        self.ui.cmbPrepOptions.addItems(['Manual', 'Abdominal CT', 'Lung CT', 'Brain CT', 'Mediastinum CT', 'MR'])
        self.ui.cmbPrepOptions.currentTextChanged.connect(lambda new_text: self.setManualPreprocessVis(new_text == 'Manual'))
        self.ui.pbApplyPrep.connect('clicked(bool)', lambda: self.logic.applyPreprocess(self.ui.cmbPrepOptions.currentText, self.ui.sldWinLevel.value, self.ui.sldWinWidth.value))

        self.ui.cmbSlicerIdx.addItems(['Select ROI on the middle slice', 'Select ROI on the first frame'])
        self.ui.cmbSlicerIdx.currentTextChanged.connect(lambda new_text: self.ui.btnMiddleSlice.setText('Segment Middle Slice' if new_text == 'Select ROI on the middle slice' else  'Segment First Frame'))

        self.checkpoint_list = {
            'Latest': 'MedSAM2_latest.pt',
            'Lesions CT scan': 'MedSAM2_CTLesion.pt',
            'Liver lesions MRI': 'MedSAM2_MRI_LiverLesion.pt',
            'Heart ultra sound': 'MedSAM2_US_Heart.pt',
            'Base model': 'MedSAM2_2411.pt'
        }
        self.ui.cmbCheckpoint.addItems(list(self.checkpoint_list.keys()))
        
        # Setting icons
        from PythonQt.QtGui import QIcon
        iconsPath = os.path.join(os.path.dirname(__file__), 'Resources/Icons')
        self.ui.pbApplyPrep.setIcon(QIcon(os.path.join(iconsPath, 'verify.png')))
        self.ui.btnStart.setIcon(QIcon(os.path.join(iconsPath, 'start.png')))
        self.ui.btnEnd.setIcon(QIcon(os.path.join(iconsPath, 'the-end.png')))
        self.ui.btnROI.setIcon(QIcon(os.path.join(iconsPath, 'bounding-box.png')))
        self.ui.btnMiddleSlice.setIcon(QIcon(os.path.join(iconsPath, 'target.png')))
        self.ui.btnRefine.setIcon(QIcon(os.path.join(iconsPath, 'performance.png')))
        self.ui.btnSegment.setIcon(QIcon(os.path.join(iconsPath, 'body-scan.png')))
        self.ui.btnRefine3D.setIcon(QIcon(os.path.join(iconsPath, 'performance.png')))
        self.ui.btnAddPoint.setIcon(QIcon(os.path.join(iconsPath, 'add-selection.png')))
        self.ui.btnSubtractPoint.setIcon(QIcon(os.path.join(iconsPath, 'sub-selection.png')))
        self.ui.btnImprove.setIcon(QIcon(os.path.join(iconsPath, 'continuous-improvement.png')))

        # Buttons
        self.ui.btnStart.connect("clicked(bool)", lambda: self.setROIboundary(lower=True))
        self.ui.btnEnd.connect("clicked(bool)", lambda: self.setROIboundary(lower=False))
        self.ui.btnROI.connect("clicked(bool)", lambda: self.drawBBox(prefix='ROI'))
        self.ui.btnMiddleSlice.connect("clicked(bool)", self.logic.getMiddleMask)
        self.ui.btnRefine.connect("clicked(bool)", self.logic.refineMiddleMask)
        self.ui.btnSegment.connect("clicked(bool)", self.logic.segment)
        self.ui.btnRefine3D.connect("clicked(bool)", self.logic.refineMiddleMask)
        self.ui.btnAddPoint.connect("clicked(bool)", lambda: self.addPoint(prefix='addition'))
        self.ui.btnSubtractPoint.connect("clicked(bool)", lambda: self.addPoint(prefix='subtraction'))
        self.ui.btnImprove.connect("clicked(bool)", lambda: self.logic.improveResult())

        self.ui.CollapsibleButton_5.setVisible(False)
        self.ui.btnAddPoint.setVisible(False)
        self.ui.btnSubtractPoint.setVisible(False)
        self.ui.btnImprove.setVisible(False)

        self.initializeParameterNode()
    
    def setManualPreprocessVis(self, visible):
        self.ui.lblLevel.setVisible(visible)
        self.ui.lblWidth.setVisible(visible)
        self.ui.sldWinLevel.setVisible(visible)
        self.ui.sldWinWidth.setVisible(visible)

    def cleanup(self) -> None:
        self.removeObservers()

    def enter(self) -> None:
        self.initializeParameterNode()

    def exit(self) -> None:
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None

    def onSceneStartClose(self, caller, event) -> None:
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        self.setParameterNode(self.logic.getParameterNode())
        if not self._parameterNode.inputVolume:
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.inputVolume = firstVolumeNode

    def setParameterNode(self, inputParameterNode: Optional[MedSAM2ParameterNode]) -> None:
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
    
    def setROIboundary(self, lower):
        if self.logic.boundaries is None:
            self.logic.boundaries = [None, None]
        curr_slice = slicer.app.layoutManager().sliceWidget("Red").sliceLogic().GetSliceOffset()
        self.logic.boundaries[int(not lower)] = curr_slice

        if None not in self.logic.boundaries:
            slice_idx = sum(self.logic.boundaries)/2 if self.ui.cmbSlicerIdx.currentText == 'Select ROI on the middle slice' else min(self.logic.boundaries)
            slicer.app.layoutManager().sliceWidget("Red").sliceLogic().SetSliceOffset(slice_idx)

        print(self.logic.boundaries)
    
    def drawBBox(self, prefix=''):
        planeNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsROINode', prefix).GetID()
        selectionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLSelectionNodeSingleton")
        selectionNode.SetReferenceActivePlaceNodeID(planeNode)
        interactionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLInteractionNodeSingleton")
        placeModePersistence = 0
        interactionNode.SetPlaceModePersistence(placeModePersistence)
        interactionNode.SetCurrentInteractionMode(1)

        slicer.mrmlScene.GetNodeByID(planeNode).GetDisplayNode().SetGlyphScale(0.5)
        slicer.mrmlScene.GetNodeByID(planeNode).GetDisplayNode().SetInteractionHandleScale(1)
    
    def addPoint(self, prefix=''):
        planeNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode', prefix).GetID()
        selectionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLSelectionNodeSingleton")
        selectionNode.SetReferenceActivePlaceNodeID(planeNode)
        interactionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLInteractionNodeSingleton")
        placeModePersistence = 0
        interactionNode.SetPlaceModePersistence(placeModePersistence)
        interactionNode.SetCurrentInteractionMode(1)

        slicer.mrmlScene.GetNodeByID(planeNode).GetDisplayNode().SetGlyphScale(0.5)
        slicer.mrmlScene.GetNodeByID(planeNode).GetDisplayNode().SetInteractionHandleScale(1)


#
# MedSAM2Logic
#

class MedSAM2Logic(ScriptedLoadableModuleLogic):

    boundaries = None
    volume_node = None
    image_data = None
    widget = None
    middleMaskNode = None
    allSegmentsNode = None
    cachedBoundaries = None
    lastSegmentLabel = None

    def __init__(self) -> None:
        ScriptedLoadableModuleLogic.__init__(self)

    def getParameterNode(self):
        return MedSAM2ParameterNode(super().getParameterNode())
    
    def captureImage(self):
        self.volume_node = slicer.util.getNodesByClass('vtkMRMLScalarVolumeNode')[0]
        if self.volume_node.GetNodeTagName() == 'LabelMapVolume': 
            outputvolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", self.volume_node.GetName())
            sef = slicer.modules.volumes.logic().CreateScalarVolumeFromVolume(slicer.mrmlScene, outputvolume, self.volume_node)
            slicer.mrmlScene.RemoveNode(self.volume_node)

            appLogic = slicer.app.applicationLogic()
            selectionNode = appLogic.GetSelectionNode()
            selectionNode.SetActiveVolumeID(sef.GetID())
            appLogic.PropagateVolumeSelection()
            self.volume_node = sef

        self.image_data = slicer.util.arrayFromVolume(self.volume_node)  
    
    def get_bounding_box(self, make2d=False):
        self.captureImage()
        
        # Grab any existing ROIs or Points
        roiNodes = list(slicer.util.getNodesByClass('vtkMRMLMarkupsROINode'))
        pointNodes = list(slicer.util.getNodesByClass('vtkMRMLMarkupsFiducialNode'))

        # Set absolute defaults to guarantee we NEVER return None
        max_z = self.image_data.shape[0] - 1
        bboxes = []
        zrange = [0, max_z]
        slice_idx = 0

        # 1. Early exit if nothing is drawn
        if len(roiNodes) == 0 and len(pointNodes) == 0:
            return slice_idx, bboxes, zrange

        # 2. Setup Coordinate Transforms (RAS to IJK)
        transformRasToVolumeRas = vtk.vtkGeneralTransform()
        slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(None, self.volume_node.GetParentTransformNode(), transformRasToVolumeRas)
        
        volumeRasToIjk = vtk.vtkMatrix4x4()
        self.volume_node.GetRASToIJKMatrix(volumeRasToIjk)

        # Inline helper so we don't need external class methods
        def ras_to_ijk(ras_point):
            vol_ras = transformRasToVolumeRas.TransformPoint(ras_point)
            ijk_point = [0, 0, 0, 1]
            volumeRasToIjk.MultiplyPoint(np.append(vol_ras, 1.0), ijk_point)
            return [int(round(c)) for c in ijk_point[0:3]]

        boundaries = [0,0] if self.boundaries is None or None in self.boundaries else self.boundaries

        # Variables to track the highest and lowest Z slice used across all markers
        global_zmin = max_z
        global_zmax = 0

        # 3. Process Bounding Boxes (if any)
        for roiNode in roiNodes:
            if make2d:
                roi_size = roiNode.GetSize()
                roiNode.SetSize(roi_size[0], roi_size[1], 1)
                roi_center = list(roiNode.GetCenter())
                slice_offset = slicer.app.layoutManager().sliceWidget("Red").sliceLogic().GetSliceOffset()
                roiNode.SetCenter([roi_center[0], roi_center[1], slice_offset])

            bounds = np.zeros(6)
            roiNode.GetBounds(bounds)
            
            p1 = [bounds[0], bounds[2], min(boundaries)]
            p2 = [bounds[1], bounds[3], max(boundaries)]
            
            ijk_p1 = ras_to_ijk(p1)
            ijk_p2 = ras_to_ijk(p2)
            
            xmin = min(ijk_p1[0], ijk_p2[0])
            ymin = min(ijk_p1[1], ijk_p2[1])
            xmax = max(ijk_p1[0], ijk_p2[0])
            ymax = max(ijk_p1[1], ijk_p2[1])
            
            zmin = min(ijk_p1[2], ijk_p2[2])
            zmax = max(ijk_p1[2], ijk_p2[2])
            
            bboxes.append(np.array([xmin, ymin, xmax, ymax]))
            
            global_zmin = min(global_zmin, zmin)
            global_zmax = max(global_zmax, zmax)

        # 4. Process Fiducial Points (if any)
        for pointNode in pointNodes:
            for i in range(pointNode.GetNumberOfControlPoints()):
                ras_p = [0.0, 0.0, 0.0]
                pointNode.GetNthControlPointPosition(i, ras_p)
                
                ijk_p = ras_to_ijk(ras_p)
                
                # Add 15px padding to make a 30x30 virtual bounding box for MedSAM2
                padding = 15
                xmin = ijk_p[0] - padding
                ymin = ijk_p[1] - padding
                xmax = ijk_p[0] + padding
                ymax = ijk_p[1] + padding
                
                bboxes.append(np.array([xmin, ymin, xmax, ymax]))
                
                # Update global bounds with the point's slice location
                global_zmin = min(global_zmin, ijk_p[2])
                global_zmax = max(global_zmax, ijk_p[2])

        # 5. Finalize Z-Range
        if len(pointNodes) > 0:
            # If using points, give MedSAM2 freedom to propagate through the whole volume
            zrange = [0, max_z]
        else:
            # If only boxes, lock it to the box boundaries
            zrange = [max(0, global_zmin), min(global_zmax, max_z)]

        # 6. Calculate the starting slice (slice_idx)
        if self.widget.ui.cmbSlicerIdx.currentText == 'Select ROI on the middle slice':
            slice_idx = int((global_zmin + global_zmax) / 2)
        else:
            slice_idx = int(global_zmin)
            
        # Hard clamp to ensure it never throws an out-of-bounds error
        slice_idx = max(0, min(slice_idx, max_z))

        return slice_idx, bboxes, zrange
    
    def get_point_coords(self):
        self.captureImage()
        pointNodes = slicer.util.getNodesByClass('vtkMRMLMarkupsFiducialNode')

        transformRasToVolumeRas = vtk.vtkGeneralTransform()
        slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(None, self.volume_node.GetParentTransformNode(), transformRasToVolumeRas)
        
        point_list = {}
        for pointNode in pointNodes:
            bounds = np.zeros(6)
            pointNode.GetBounds(bounds)
            curr_point = bounds[::2].copy()

            point_VolumeRas = transformRasToVolumeRas.TransformPoint(curr_point)
            volumeRasToIjk = vtk.vtkMatrix4x4()
            self.volume_node.GetRASToIJKMatrix(volumeRasToIjk)
            point_Ijk = [0, 0, 0, 1]
            volumeRasToIjk.MultiplyPoint(np.append(point_VolumeRas,1.0), point_Ijk)
            point_Ijk = [ int(round(c)) for c in point_Ijk[0:3] ]
            
            point_list[pointNode.GetID()] = point_Ijk

        return point_list
    
    def run_on_background(self, target, args, title):
        self.progressbar = slicer.util.createProgressDialog(autoClose=False)
        self.progressbar.minimum = 0
        self.progressbar.maximum = 0
        self.progressbar.setLabelText(title)
        
        job_event = threading.Event()
        paral_thread = threading.Thread(target=target, args=(*args, job_event,))
        paral_thread.start()
        while not job_event.is_set():
            slicer.app.processEvents()
        paral_thread.join()

        self.progressbar.close()
    
    def segment_helper(self, img_path, json_payload, result_path, ip, port, job_event):
        config, checkpoint = self.getConfigCheckpoint()
        self.progressbar.setLabelText(' Segmenting (this may take a while)... ')
        
        url = f'http://{ip}:{port}/segment'
        data = {
            'checkpoint': checkpoint,
            'config': config,
            'bboxes': json_payload,
            'propagate': True
        }
        
        try:
            with open(img_path, 'rb') as f:
                # files = {'file': (os.path.basename(img_path), f, 'application/gzip')}
                files = {'file': (os.path.basename(img_path), f, 'application/octet-stream')}
                response = requests.post(url, data=data, files=files)
                
            if response.status_code == 200:
                with open(result_path, 'wb') as f:
                    f.write(response.content)
            else:
                print("Server Error:", response.text)
        except Exception as e:
            print("Request Failed:", e)
            
        job_event.set()
    
    def showSegmentation(self, result_path, set_middle_mask=False, improve_previous=False):
        if self.allSegmentsNode is None:
            self.allSegmentsNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")

        current_seg_group = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode") if set_middle_mask else self.allSegmentsNode
        current_seg_group.SetReferenceImageGeometryParameterFromVolumeNode(self.volume_node)

        # THE FIX: Directly load the returned NRRD file as a LabelMap. 
        # This forces Slicer to natively align the origin, spacing, and direction matrix!
        result_labelmap = slicer.util.loadLabelVolume(result_path, {"singleFile": True})
        
        if result_labelmap:
            # --- START DIAGNOSTICS ---
            mask_array = slicer.util.arrayFromVolume(result_labelmap)
            print("\n" + "="*40)
            print(f"DIAGNOSTICS FOR RETURNED MASK")
            print(f"File Path: {result_path}")
            print(f"Array Shape: {mask_array.shape}")
            print(f"Unique Pixel Values: {np.unique(mask_array)}")
            print(f"Total Segmented Pixels: {np.count_nonzero(mask_array)}")
            print("="*40 + "\n")
            # --- END DIAGNOSTICS ---
            slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(result_labelmap, current_seg_group)
            slicer.mrmlScene.RemoveNode(result_labelmap)

        if set_middle_mask:
            self.middleMaskNode = current_seg_group
        else:
            try:
                slicer.mrmlScene.RemoveNode(self.middleMaskNode)
            except:
                pass
                
        if improve_previous and self.lastSegmentLabel:
            self.allSegmentsNode.GetSegmentation().RemoveSegment(self.lastSegmentLabel)
        
        # Track the newest segment ID
        segmentation = current_seg_group.GetSegmentation()
        if segmentation.GetNumberOfSegments() > 0:
            self.lastSegmentLabel = segmentation.GetNthSegmentID(segmentation.GetNumberOfSegments() - 1)
            print('self.lastSegmentLabel is updated to', self.lastSegmentLabel)

    def segment(self):
        self.captureImage()
        slice_idx, bboxes, zrange = self.get_bounding_box(make2d=False)
        bboxes_list = [bbox.tolist() for bbox in bboxes]
        
        with tempfile.TemporaryDirectory() as tmpdirname:
            img_path = f"{tmpdirname}/img_data.nrrd" # Changed to NRRD
            slicer.util.saveNode(self.volume_node, img_path) 
            
            result_path = f"{tmpdirname}/result.nrrd" # Changed to NRRD
            payload = json.dumps({
                'boxes': bboxes_list, 
                'z_range': [int(zrange[0]), int(zrange[1]), int(slice_idx)]
            })
            
            self.run_on_background(
                self.segment_helper, 
                (img_path, payload, result_path, self.widget.ui.txtIP.text.strip(), self.widget.ui.txtPort.text.strip()), 
                'Segmenting Volume...'
            )

            # THE FIX: Just pass the path directly to showSegmentation
            if os.path.exists(result_path):
                self.showSegmentation(result_path)

            self.cachedBoundaries = {'bboxes': bboxes, 'zrange': zrange}
            self.widget.ui.CollapsibleButton_5.setVisible(True)

        roiNodes = slicer.util.getNodesByClass('vtkMRMLMarkupsROINode')
        for roiNode in roiNodes:
            slicer.mrmlScene.RemoveNode(roiNode)
        self.boundaries = None
    
    def middle_mask_helper(self, img_path, json_payload, result_path, ip, port, job_event):
        config, checkpoint = self.getConfigCheckpoint()
        self.progressbar.setLabelText(' Segmenting Middle Slice... ')
        
        url = f'http://{ip}:{port}/segment'
        data = {
            'checkpoint': checkpoint,
            'config': config,
            'bboxes': json_payload,
            'propagate': False
        }
        
        try:
            with open(img_path, 'rb') as f:
                # files = {'file': (os.path.basename(img_path), f, 'application/gzip')}
                files = {'file': (os.path.basename(img_path), f, 'application/octet-stream')}
                response = requests.post(url, data=data, files=files)
                
            if response.status_code == 200:
                with open(result_path, 'wb') as f:
                    f.write(response.content)
            else:
                print("Server Error:", response.text)
        except Exception as e:
            print("Request Failed:", e)
            
        job_event.set()

    def getMiddleMask(self):
        self.captureImage()
        slice_idx, bboxes, zrange = self.get_bounding_box(make2d=True)
        bboxes_list = [bbox.tolist() for bbox in bboxes]
        
        with tempfile.TemporaryDirectory() as tmpdirname:
            img_path = f"{tmpdirname}/img_data.nrrd" # Changed to NRRD
            slicer.util.saveNode(self.volume_node, img_path)
            
            result_path = f"{tmpdirname}/result.nrrd" # Changed to NRRD
            payload = json.dumps({
                'boxes': bboxes_list, 
                'z_range': [int(zrange[0]), int(zrange[1]), int(slice_idx)]
            })
            
            self.run_on_background(
                self.middle_mask_helper, 
                (img_path, payload, result_path, self.widget.ui.txtIP.text.strip(), self.widget.ui.txtPort.text.strip()), 
                'Segmenting Middle Slice...'
            )
            
            if os.path.exists(result_path):
                self.showSegmentation(result_path, set_middle_mask=True)
        
        roiNodes = slicer.util.getNodesByClass('vtkMRMLMarkupsROINode')
        for roiNode in roiNodes:
            roiNode.SetDisplayVisibility(False)
    
    def refineMiddleMask(self):
        slicer.util.selectModule("SegmentEditor") 
    
    def preprocess_CT(self, win_level=40.0, win_width=400.0):
        self.captureImage()
        lower_bound, upper_bound = win_level - win_width/2, win_level + win_width/2
        image_data_pre = np.clip(self.image_data, lower_bound, upper_bound)
        image_data_pre = (image_data_pre - np.min(image_data_pre))/(np.max(image_data_pre)-np.min(image_data_pre))*255.0
        image_data_pre = np.uint8(image_data_pre)

        self.volume_node.GetDisplayNode().SetAutoWindowLevel(False)
        self.volume_node.GetDisplayNode().SetWindowLevelMinMax(0, 255)
        
        return image_data_pre
    
    def preprocess_MR(self, lower_percent=0.5, upper_percent=99.5):
        self.captureImage()
        lower_bound, upper_bound = np.percentile(self.image_data[self.image_data > 0], lower_percent), np.percentile(self.image_data[self.image_data > 0], upper_percent)
        image_data_pre = np.clip(self.image_data, lower_bound, upper_bound)
        image_data_pre = (image_data_pre - np.min(image_data_pre))/(np.max(image_data_pre)-np.min(image_data_pre))*255.0
        image_data_pre = np.uint8(image_data_pre)

        self.volume_node.GetDisplayNode().SetAutoWindowLevel(False)
        self.volume_node.GetDisplayNode().SetWindowLevelMinMax(0, 255)

        return image_data_pre
    
    def updateImage(self, new_image):
        self.image_data[:,:,:] = new_image
        slicer.util.arrayFromVolumeModified(self.volume_node)
    
    def applyPreprocess(self, method, win_level, win_width):
        if method == 'MR':
            prep_img = self.preprocess_MR()
        elif method == 'Manual':
            prep_img = self.preprocess_CT(win_level = win_level, win_width = win_width)
        else:
            conversion = {
                'Abdominal CT': (400.0, 40.0),
                'Lung CT': (1500.0, -600.0),
                'Brain CT': (80.0, 40.0),
                'Mediastinum CT': (350.0, 50.0),
            }
            ww, wl = conversion[method]
            prep_img = self.preprocess_CT(win_level = wl, win_width = ww)

        self.updateImage(prep_img)
    
    def getConfigCheckpoint(self):
        if self.widget.ui.pathConfig.currentPath == '':
            config = 'MedSAM2_tiny512.yaml'
        else:
            config = 'custom_' + os.path.basename(self.widget.ui.pathConfig.currentPath)
        
        if self.widget.ui.pathModel.currentPath == '':
            checkpoint = self.widget.checkpoint_list[self.widget.ui.cmbCheckpoint.currentText]
        else:
            model_name = os.path.basename(self.widget.ui.pathModel.currentPath).split('.')[0]
            checkpoint = os.path.join(model_name, os.path.basename(self.widget.ui.pathModel.currentPath))
        
        return config, checkpoint
    
    def improve_helper(self, img_path, json_payload, result_path, ip, port, job_event):
        self.progressbar.setLabelText(' Improving Segmentation... ')
        
        url = f'http://{ip}:{port}/improve'
        data = {
            'points': json_payload
        }
        
        try:
            with open(img_path, 'rb') as f:
                # files = {'file': (os.path.basename(img_path), f, 'application/gzip')}
                files = {'file': (os.path.basename(img_path), f, 'application/octet-stream')}
                response = requests.post(url, data=data, files=files)
                
            if response.status_code == 200:
                with open(result_path, 'wb') as f:
                    f.write(response.content)
            else:
                print("Server Error:", response.text)
        except Exception as e:
            print("Request Failed:", e)
            
        job_event.set()

    def improveResult(self):
        point_list = self.get_point_coords()
        points_partition = {'addition': [], 'subtraction': []}
        for point_name in point_list:
            point_type = 'addition' if 'addition' in slicer.util.getNode(point_name).GetName() else 'subtraction'
            points_partition[point_type].append(point_list[point_name])

        with tempfile.TemporaryDirectory() as tmpdirname:
            img_path = f"{tmpdirname}/img_data.nrrd" # Changed to NRRD
            slicer.util.saveNode(self.volume_node, img_path)
            
            result_path = f"{tmpdirname}/result.nrrd" # Changed to NRRD
            
            payload = json.dumps({
                'bboxes': [bbox.tolist() for bbox in self.cachedBoundaries['bboxes']],
                'zrange': [int(self.cachedBoundaries['zrange'][0]), int(self.cachedBoundaries['zrange'][1])],
                'points_addition': points_partition['addition'],
                'points_subtraction': points_partition['subtraction']
            })
            
            self.run_on_background(
                self.improve_helper, 
                (img_path, payload, result_path, self.widget.ui.txtIP.text.strip(), self.widget.ui.txtPort.text.strip()), 
                'Improving Segmentation...'
            )
            
            if os.path.exists(result_path):
                self.showSegmentation(result_path, improve_previous=True)
        
        pointNodes = slicer.util.getNodesByClass('vtkMRMLMarkupsFiducialNode')
        for pointNode in pointNodes:
            slicer.mrmlScene.RemoveNode(pointNode)

#
# MedSAM2Test
#

class MedSAM2Test(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_MedSAM21()

    def test_MedSAM21(self):
        self.delayDisplay("Starting the test")
        self.delayDisplay("Test passed")