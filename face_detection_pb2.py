# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: face_detection.proto
# Protobuf Python Version: 5.28.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    5,
    28,
    1,
    '',
    'face_detection.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x14\x66\x61\x63\x65_detection.proto\x12\x0e\x66\x61\x63\x65_detection\"\x1b\n\nImageChunk\x12\r\n\x05image\x18\x01 \x01(\x0c\";\n\x16\x46\x61\x63\x65ValidationResponse\x12\x10\n\x08is_valid\x18\x01 \x01(\x08\x12\x0f\n\x07message\x18\x02 \x01(\t2l\n\x14\x46\x61\x63\x65\x44\x65tectionService\x12T\n\x0cValidateFace\x12\x1a.face_detection.ImageChunk\x1a&.face_detection.FaceValidationResponse(\x01\x42\x1f\n\x1b\x63om.face.faceanalyzer.protoP\x01\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'face_detection_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  _globals['DESCRIPTOR']._loaded_options = None
  _globals['DESCRIPTOR']._serialized_options = b'\n\033com.face.faceanalyzer.protoP\001'
  _globals['_IMAGECHUNK']._serialized_start=40
  _globals['_IMAGECHUNK']._serialized_end=67
  _globals['_FACEVALIDATIONRESPONSE']._serialized_start=69
  _globals['_FACEVALIDATIONRESPONSE']._serialized_end=128
  _globals['_FACEDETECTIONSERVICE']._serialized_start=130
  _globals['_FACEDETECTIONSERVICE']._serialized_end=238
# @@protoc_insertion_point(module_scope)
