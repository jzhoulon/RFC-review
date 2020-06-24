# **Pluggable device for Tensorflow**

<table>
  <tr>
    <td>Status</td>
    <td>(Proposed / Accepted / Implemented / Obsolete)</td>
  </tr>
  <tr>
    <td>RFC #</td>
    <td>NNN (update when you have community PR #)</td>
  </tr>
  <tr>
    <td>Author(s)</td>
    <td>Zhoulong Jiang, Yiqiang Li, Eric Lin, Jianhui Li</td>
  </tr>
  <tr>
    <td>Sponsor</td>
    <td>Anna Revinskaya (annarev@google.com)</td>
  </tr>
  <tr>
    <td>Updated</td>
    <td>2020-06-19</td>
  </tr>
</table>


## **Objective**

Implement a pluggable device mechanism which allows to run existing tensorflow programs on a new device without user changing most of the code.  Users only need to install a shared library in a specified directory, and the mechanism is able to discover and plug in the capabilities offered by the library. 

This RFC is based on the Modular Tensorflow  [RFC](https://github.com/tensorflow/community/pull/77), which aims to extend the Tensorflow design to plugin capabilities like adding a new device support.  The modular device interface is based on StreamExecutor C API [RFC](https://github.com/tensorflow/community/pull/257). 

## **Motivation**

When extending Tensorflow to support a new device, one needs to modify tensorflow code and maintain a special tensorflow build for the new device. Modular Tensorflow RFC provides a mechanism which adds the device support, build in a separate shared library, at runtime.  This RFC further describes how tensorflow automatically discovers these device libraries and adds them to tensorflow.  

The pluggable device discovery and initialization is transparent to end users. As long as the device plugin libraries follow the interface described in this RFC, it can be plugged to tensorflow proper and enable tensorflow to run existing tensorflow programs on a new device. 

## **User Benefit**

This allows tensorflow to transparently run tensorflow programs on new devices, as long as users set up the system properly to include device plugin libraries. 

## **Design Proposal**

### Design Overview

This RFC describes the mechanism of extending the tensorflow device class hierarchy to add pluggable device as shown in diagram 1:
<div align="center">
<img src="https://github.com/jzhoulon/RFC-review/blob/master/design_overview.png" />
</div>

* `PluggableDevice` is a virtual device defined in Tensorflow proper which inherits [LocalDevice](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/local_device.h).It is built on top of  StreamExecutor C++ interface to manage `PluggableDevice`’s key abstractions like stream, memory and event.

* `PluggableDeviceExecutor` implements [StreamExecutor](https://github.com/tensorflow/tensorflow/blob/e5023a1738cce7efcdf9d87863b85c80ab2f8c9e/tensorflow/stream_executor/stream_executor_pimpl.h#L73) and is built on top of StreamExecutor C API (addressed in [RFC](https://github.com/tensorflow/community/pull/257)). 

* `PluggableDevice Backend` is inside the TF plugin, which implements StreamExecutor C API and registers its platform to the Tensorflow proper when the plugin’s shared library is loaded. 

The pluggable device mechanism contains device discovery and creation process which creates a `PluggableDevice` object and `PluggableDeviceExecutor` object for each PluggableDevice Backend. 

With the RFC, existing tensorflow GPU programs can run on a plugged device without the user changing the code. The diagram 2 describes the workflow of Tensorflow with device plugin, it shows how a simple GPU program runs on the pluggable device.
<div align="center">
<img src="https://github.com/jzhoulon/RFC-review/blob/master/gpu_example.png">
</div>

### Device Discovery

Upon initialization of Tensorflow, it uses platform independent `LoadLibrary()` to load the dynamic library. The PluggableDevice Backend plugin library should be installed to default plugin directory "…python_dir.../site-packages/tensorflow-plugins". The modular tensorflow [RFC](https://github.com/tensorflow/community/pull/77) describes the process loading plugins. 

During the plugin library initialization, it calls the `SE_ReigsterPlatform()` API to register the stream executor platform (`SE_Platform` struct) to Tensorflow proper. The `SE_ReigsterPlatform()` API is a callback API, part of StreamExecutor C API, which passes necessary information to Tensorflow proper to instantiate a stream executor platform (`se::platform` class) and register to a global object `se::MultiPlatformManager`. 
The stream executor platform must be registered with the name "PluggableDevice".  
See below code which is an example of registering a PluggableDevice platform with StreamExecutor C API:
```cpp
void RegisterPluggableDevicePlatform() {
  static plugin_id_value = 123;
  SE_PlatformId id;
  id.id = &plugin_id_value;
  int visible_device_count = get_plugin_device_count;
  SE_Platform* custom_platform = SE_NewPlatform(
     id, visible_device_count,
     create_device, create_stream_executor,
     delete_device, delete_stream_executor);
  TF_Status* status = TF_NewStatus();
  std::string name = "PluggableDevice";
  SE_RegisterPlatform(
     name.c_str(), name.size(),
     custom_platform,
     status);
}

```
Use static initialization to register the new platform:
```cpp
static bool IsPluggableDevicePlatformRegistered = []() {
 RegisterPluggablePlatform();
 return true;
}();

```

### Device Creation

`PluggableDeviceFactory` is introduced to create the `PluggableDevice`, following the [LocalDevice](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/local_device.h) design pattern. To support existing GPU programs run on a new device without user changing the most of the code , `PluggableDeviceFactory` is registered as "GPU" device name and given higher priority than the default GPU. 
```cpp
   REGISTER_LOCAL_DEVICE_FACTORY("GPU",PluggableDeviceFactory, 220); // plugged GPU
   REGISTER_LOCAL_DEVICE_FACTORY("GPU", GPUDeviceFactory, 210);//default GPU
```
When a session is created, `PluggableDeviceFactory` creates a `PluggableDevice` object for the plugin device. During the initialization of the `PluggableDevice`, a global object `se::MultiPlatformManager` will find its `se::platform` through its platform name: "PluggableDevice”,  then stream executor platform (`se::platform`) further creates a StreamExecutor object containing a `PluggableDeviceExecutor`, and multiple stream objects(a computation stream and several memory copy streams) supporting the StreamExecutor objects. 

See below the example code which creates the `PluggableDeviceExecutor` using the information registered during plugin library initialization. 

The section below shows some pseudo code to introduce some changes to the Tensorflow proper and what needs to be implemented in the plugin for the pluggable device creation. The implementation is based on StreamExecutor C API [RFC](https://github.com/tensorflow/community/pull/257). 

1. `PluggableDeviceFactory` creates and initializes a set of pluggable devices when the session is created.  
```cpp
   PluggableDeviceFactory::CreateDevices(SessionOptions& options, const string& name_prefix, std::vector<std::unique_ptr<Device>>* devices) {
     for (int i = 0; i < options.device_count(); i++) {
      PluggableDevice pluggable_device = CreatePluggableDevice(options,i); //set allocator
      pluggable_device->Init(options);
      devices.push_back(std::move(pluggable_device));
     }
   }
```

2. `PluggableDevice` object binds a StreamExecutor and creates a set of Streams during the initialization.Streams include one compute stream and several memory copy streams.
```cpp
   void PluggableDevice::Init(SessionOption& options) {  
     se::Platform* platform= se::MultiPlatformManager::PlatformWithName("PluggableDevice");
     stream_executor_ = platform->ExecutorForDevice(pluggable_dev_id_);
     compute_stream_ = new se::Stream(stream_executor_);
     compute_stream_->Init();
     host_to_device_stream_ = new se::Stream(stream_executor_);
     host_to_device_stream_->Init();
     ...
   }  // create StreamExecutor
```
3. `PluggableDevicePlatform` is responsible for the StreamExecutor creation. It creates an `SE_StreamExecutor` and `SE_Device` object through create_stream_executor and create_device function handle which are registered in the `SE_Platform`. Then `PluggableDeviceExecutor` is then constructed with `SE_StreamExecutor` and `SE_Device` object.   
```cpp
   StreamExecutor* PluggableDevicePlaform::ExeutorForDevice(int device_id） {
    auto config = get_plugin_config(device_id);
    SE_Options* se_option = get_se_option(device_id);
    SE_StreamExecutor* se= platform_->create_stream_executor();
    SE_Device* sd = platform_->create_device(se_options)
    auto executor = absl::make_unique<StreamExecutor>(this, absl::make_unique<PluggableDeviceExecutor>(config, se, sd));
    return std::move(executor);
   }
```
**Tensorflow Proper**

Tensorflow proper needs to be extended to support a new virtual device (`PluggableDevice`) to represent a set of new third-party devices and a new stream executor platform (`PluggableDevicePlatform`) to create the device and related resources with the information registered from plugin. 

Two sets of classes need to be defined in Tensorflow proper. 
* Set 1: `PluggableDevice` related classes 
   * class `PluggableDevice`: a virtual device represents a set of new third-party devices, it has a new device type named "PluggableDevice"/DEVICE_PLUGGABLE.
   * class `PluggableDeviceFactory`: a device factory to create the PluggableDevice
   * class `PluggableDeviceBFCAllocator`: a PluggableDevice memory allocator that implements a ‘best fit with coalescing’ algorithm.
   * class `PluggableDeviceAllocator`: an allocator that wraps a PluggableDevice allocator.
   * class `PluggableDeviceHostAllocator`: allocator for pinned CPU RAM that is made known to PluggableDevice for the purpose of efficient DMA with PluggableDevice.
   * class `PluggableDeviceEventMgr`: an object to keep track of pending Events in the StreamExecutor streams.
   * class `PluggableDeviceContext`: a wrapper of pluggable device specific context that can be passed to OpKernels.
* Set 2: `PluggableDevicePlatform` related classes 
   * class `PluggableDevicePlatform`: PluggableDevice-specific platform, its platform name is "PluggableDevice", it contains a C struct: SE_Platform* platform_ which is its internal implementation and as the C interface registered by device plugin.
   * class `PluggableDeviceExecutor`: PluggableDevice-platform implementation of the platform-agnostic StreamExecutorInterface, it contains C structs: SE_StreamExecutor* executor_ and SE_Device* device_ whose member can be accessed in both Tensorflow proper and device plugins.
   * class `PluggableDeviceStream`: wraps a StreamHandle in order to satisfy the platform-independent StreamInterface. It returns SE_Stream which is treated as an opaque type to Tensorflow,  whose structure is created by the device plugin.  
   * class `PluggableDeviceTimer`: wraps an opaque handle: SE_Timer to satisfy the platform-independent TimerInterface.
   * class `PluggableDeviceEvent`: wraps an opaque handle: SE_Event to satisfy the platform-independent EventInterface.

**Plugin**

Plugins need to implement and register the StreamExecutor C API defined in the Tensorflow proper. 
*  `SE_StreamExecutor` is defined as struct in the C API, both sides(Tensorflow proper and plugins) can access its members. Plugin creates the SE_StreamExecutor and registers its C API implementations to the SE_StreamExecutor.  
```cpp
   SE_StreamExecutor* create_stream_executor() {
     SE_StreamExecutor* se_nfs = new SE_StreamExecutor();
     se->memcpy_from_host = my_device_memory_from_host_function;
     se->allocate = my_allocate_function;
     …
   }//Init device
```
* `SE_Device` is defined as struct in the C API, both sides(Tensorflow proper and plugins) can access its members. Plugin creates the SE_Device and fills its device opaque handle and device name to the SE_Device.
```cpp
  SE_Device* create_device(SE_Options* options, TF_Status* status) {
    SE_Device* se = new SE_Device();
    se->device_handle = get_my_device_handle();
    ...
    return se;
  }
```
* `SE_Stream` is defined in plugin and treated as an opaque struct in Tensorflow proper. 
```cpp
  void create_stream(SE_Device* executor, SE_Stream* stream, TF_Status*) {
    *stream = new SE_Stream_st();
    (*stream)->stream_handle = create_my_stream_handle(executor);
    ..
  }
```

### PluggableDevice kernel registration

This RFC shows an example of registering kernels for PluggableDevice. Kernel and op registration and implementation API is addressed in a separate [RFC](https://github.com/tensorflow/community/blob/master/rfcs/20190814-kernel-and-op-registration.md). 

Tensorflow proper defines a new device_type named DEVICE_PLUGGABLE for PluggableDevice.This device_type is used for the kernel registration and dispatch. Plugin needs to register its kernel implementation with DEVICE_PLUGGABLE type.
```cpp
void InitPlugin() {
  TF_KernelBuilder* builder = TF_NewKernelBuilder(/*op_name*/"Convolution", DEVICE_PLUGGABLE,
      &Conv_Create, &Conv_Compute, &Conv_Delete);
  TF_Status* status = TF_NewStatus();
  TF_RegisterKernelBuilder(/*kernel_name*/"Convolution", builder, status);
  if (TF_GetCode(status) != TF_OK) { /* handle errors */ }
  TF_DeleteStatus(status);
}
```
### Using stream inside PluggableDevice kernel

The following code shows a convolution kernel implementation using the stream handle. The streams are created during the pluggable device creation. The placer decides which device to use for each OP in the graph. Then the streams associated with the device are used to construct the OpKernelContext for the op computation during the graph execution.
```cpp
void Conv_Compute(TF_OpKernelContext*) {
  TF_GetInput(context, input_index, &input, &status);
  TF_GetInput(context, filter_index, &filter, &status);
  auto output = TF_AllocateOutput(context, output_index, TF_Float32, dims, num_dims, len, status);
  SE_Stream se_stream = TF_GetStream(TF_OpKernelContext);
  auto native_stream = static_cast<native_stream_type>(se_stream->stream_handle);
  my_conv_impl(input, filter, output, native_stream);
}
```
Kernel and op registration and implementation API [RFC](https://github.com/tensorflow/community/blob/master/rfcs/20190814-kernel-and-op-registration.md) needs to be extended to retrieve streams/device context from the TF_OpKernelContext, besides inputs and outputs. 

### **Alternatives Considered**

* Without this RFC, end users need to change the python code to import the third-party device plugin. 

* Without this RFC, the third-party device vendor may implement the LocalDevice interface, which is not a C API interface and may interact with potential C++ ABI incompatibility issues.  

### **Performance Implications**

* We don’t expect performance impact due to this RFC. The functions described by this RFC are realized at the initialization stage. 

### **Dependencies**

* This RFC doesn’t add new dependencies to external libraries. 

* It depends on three modular Tensorflow related RFC 

    * Modular Tensorflow  [RFC](https://github.com/tensorflow/community/pull/77)

    * StreamExecutor C interface [RFC](https://github.com/tensorflow/community/pull/257)

    * Kernel and op registration and implementation API [RFC](https://github.com/tensorflow/community/blob/master/rfcs/20190814-kernel-and-op-registration.md) 

### **Engineering Impact**

* The impact to binary size / startup time / build time / test times are minimum. 

* The TensorFlow team will maintain this code. 

### **Platforms and Environments**

* The pluggable device mechanism is based on loadlibrary() so should work on all the platforms supported by loadlibrary. The other enhancement to tensorflow proper is platform independent.

### **Best Practices**

* This works with Modular Tensorflow which will be the only way to integrate new third-party devices to the current Tensorflow stack. 

### **Compatibility**

The RFC promotes the current Tensorflow ecosystem as it supports plugging new devices to Tensorflow.  

We don't expect this proposal to impact with other parts of the Tensorflow ecosystem. It doesn't support TFLite. It should not impede distribution strategies and would not interact with tf.fuction and SaveModel.  

