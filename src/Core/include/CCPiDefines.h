/*
This work is part of the Core Imaging Library developed by
Visual Analytics and Imaging System Group of the Science Technology
Facilities Council, STFC

Copyright 2017 Srikanth Nagella, Edoardo Pasca, Daniil Kazantsev

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef CCPIDEFINES_H
#define CCPIDEFINES_H

#if defined(_WIN32) || defined(__WIN32__)
  #if defined(CCPiCore_EXPORTS) || defined(CCPiNexusWidget_EXPORTS) || defined(ContourTreeSegmentation_EXPORTS) || defined(ContourTree_EXPORTS)// add by CMake 
    #define  CCPI_EXPORT __declspec(dllexport)
    #define EXPIMP_TEMPLATE
  #else
    #define  CCPI_EXPORT __declspec(dllimport)
    #define EXPIMP_TEMPLATE extern
  #endif /* CCPi_EXPORTS */
#elif defined(linux) || defined(__linux) || defined(__APPLE__)
 #define CCPI_EXPORT
#endif

#endif
