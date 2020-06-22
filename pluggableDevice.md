#
# **Pluggable device for Tensorflow**

| **Status** | **(Proposed / Accepted / Implemented / Obsolete)** |
| --- | --- |
| RFC # | [NNN](https://github.com/tensorflow/community/pull/NNN) (update when you have community PR #) |
| Author(s) | Zhoulong Jiang, Yiqiang Li, Eric Lin, Jianhui Li |
| Sponsor | Anna Revinskaya (annarev@google.com) |
| Updated | 2020-06-19 |
| Obsoletes | TF-RFC it replaces, else remove this header |

## **Objective**

Implement a pluggable device mechanism which allows to run existing tensorflow programs on a new device without user changing the code. Users only need to install a dynamic library in a specified directory, and the mechanism is able to discover and plug in the capabilities offered by the library.

This RFC is based on the Modular Tensorflow RFC, which aims to extend the Tensorflow design to plugin capabilities like adding a new device support. The modular device interface is described by a separate RFC.

## **Motivation**

When extending Tensorflow to support a new device, one needs to modify tensorflow code and maintain a special tensorflow build for the new device. Modular Tensorflow RFC provides a mechanism which adds the device support, built in a separate library, at runtime. This RFC further describes how tensorflow automatically discovers these device libraries and adds them to tensorflow.

The pluggable device discovery and initialization is transparent to end users. As long as the device plugin libraries follow the interface described in this RFC, it can be plugged to tensorflow and run existing tensorflow programs targeting GPU device type.

## **User Benefit**

This allows tensorflow to transparently run tensorflow programs on new devices, as long as users set up the system properly to include device plugin libraries.

## **Design Proposal**

![](RackMultipart20200622-4-zq8joc_html_16660c9d78a2a70d.gif) **Design Overview**

The diagram 1. describes the mechanism of pluggable device.

PluggableDevice is a virtual device defined in Tensorflow proper which inherits LocalDevice.It is built on top of StreamExecutor C++ interface to manage PluggableDevice&#39;s device, stream and data movement.

PluggableDeviceExecutor is StreamExecutor&#39;s implementation and built on top of StreamExecutor C API(addressed in [RFC](https://github.com/tensorflow/community/pull/257)).

PluggableDevice Backend is part of modular TF plugin, which represents the physical device runtime. It implements StreamExecutor C API and registers its platform to the Tensorflow proper when the plugin&#39;s shared object is loaded.

The pluggable device mechanism contains device discovery and creation process which creates a PluggableDevice object and PluggableDevice Executor object for each PluggableDevice Backend.

The intention is that existing tensorflow GPU programs can run on a plugged device without the user changing the code.The diagram 2 describes the workflow of Tensorflow with device plugin, it shows how a simple GPU program runs on the pluggable device.

![](RackMultipart20200622-4-zq8joc_html_f0356ecb13863159.gif)

- **Device Discovery**

The PluggableDevice Backend is realized in a plugin library installed to default plugin directory &quot;…python\_dir.../site-packages/tensorflow-plugins&quot;. The modular tensorflow [RFC](https://github.com/tensorflow/community/pull/77) describes the process loading plugins, which uses platform independent LoadLibrary() to load the dynamic library. The plugin library implements the StreamExecutor C API as defined in the[RFC](https://github.com/tensorflow/community/pull/257) and the SE\_ReigsterPlatform() API registers the platform to a global object named MultiPlatformManager inside Tensorflow core during the load time. See below for an example of registering a new platform with StreamExecutor C API:

| ```cpp **void**** RegisterMyCustomPlatform**() {static plugin\_id\_value = 123;SE\_PlatformId id;id.id = &amp;plugin\_id\_value;int visible\_device\_count = 2;
SE\_Platform\* custom\_platform = SE\_NewPlatform(id, visible\_device\_count,create\_device, create\_stream\_executor,delete\_device, delete\_stream\_executor);
TF\_Status\* status = TF\_NewStatus();std::string name = &quot;MyCustomDevice&quot;;SE\_RegisterPlatform(name. **c\_str** (), name. **size** (),custom\_platform,status);}```
Use **static** initialization to **register** the **new** platform:
```cpp **static**** bool** IsMyCustomPlatformRegistered = []() {RegisterMyCustomPlatform();return true;}(); |
| --- |

- **Device Creation**

Following the LocalDevice design, the RFC introduced PluggableDeviceFactory, which creates the PluggableDevice. To support existing GPU programs run on a new device without user changing the code , PluggableDeviceFactory is registered as &quot;GPU&quot; device name and given higher priority than the default GPU.

REGISTER\_LOCAL\_DEVICE\_FACTORY(&quot;GPU&quot;,PluggableDeviceFactory, 220); // plugged GPU

REGISTER\_LOCAL\_DEVICE\_FACTORY(&quot;GPU&quot;, GPUDeviceFactory, 210);//default GPU

PluggableDeviceFactory creates a PluggableDevice object for the plugin device when a session is created. During the initialization of the PluggableDevice, the StreamExecutorPlatform (se::platform) further creates a StreamExecutor object containing a PluggableDeviceExecutor, and multiple stream objects(a computation stream and several memory copy streams) supporting the StreamExecutor objects.

**##Implementation**

This section shows some pseudo code to introduce some changes to the Tensorflow proper and what needs to be implemented in the plugin for the pluggable device creation. The implementation is based on [StreamExecutor C API RFC](https://github.com/tensorflow/community/pull/257)

**###Tensorflow Proper**

Tensorflow proper will add a new virtual device named PluggableDevice which represents a set of new third-party devices.Following the LocalDevice design, a set of class need to be defined in Tensorflow proper:

class PluggableDevice : a virtual device represents a set of new third-party devices

class PluggableDeviceFactory: a device factory to create the PluggableDevice

class PluggableDeviceBFCAllocator: a PluggableDevice memory allocator that implements a &#39;best fit with coalescing&#39; algorithm.

class PluggableDeviceAllocator: an allocator that wraps a PluggableDevice allocator.

class PluggableDeviceHostAllocator: allocator for pinned CPU RAM that is made known to PluggableDevice for the purpose of efficient DMA with PluggableDevice.

class PluggableDeviceEventMgr: an object to keep track of pending Events in the StreamExecutor streams.

class PluggableDeviceContext: a wrapper of pluggable device specific context that can be passed to OpKernels.

...

Tensorflow proper will add a new StreamExecutor Platform named PluggableDevicePlatform whose implementation is registered in plugin.

class PluggableDevicePlatform : PluggableDevice-specific platform, it contains a C struct: SE\_Platform\* platform\_ which is its internal implementation and as the C interface registered by device plugin.

class PluggableDeviceExecutor: PluggableDevice-platform implementation of the platform-agnostic StreamExecutorInterface, it contains C structs: SE\_StreamExecutor\* executor\_ and SE\_Device\* device\_ whose member can be accessed in both Tensorflow proper and device plugins.

class PluggableDeviceStream : wraps a StreamHandle in order to satisfy the platform-independent StreamInterface. It returns SE\_Stream which is treated as an opaque type to Tensorflow, whose structure is created by the device plugin.

class PluggableDeviceTimer : wraps an opaque handle: SE\_Timer to satisfy the platform-independent TimerInterface.

class PluggableDeviceEvent : wraps an opaque handle: SE\_Event to satisfy the platform-independent EventInterface.

...

The following pseudocode shows the process of PluggableDevice creation.

1. PluggableDeviceFactory creates and initializes a set of pluggable devices when the session is created.

| **PluggableDeviceFactory::CreateDevices** (SessionOptions&amp; options, vector\&lt;Device\*\&gt; devices) {for (int i = 0; i \&lt; options. **device\_count** (); i++) {PluggableDevice pluggable\_device = CreatePluggableDevice(options); //set allocatorpluggable\_device-\&gt; **Init** (options);devices. **push\_back** ( **std::move** (pluggable\_device));}} |
| --- |

2. PluggableDevice object will bind to a StreamExecutor and creates a set of Streams during the initialization.Streams include one compute stream and several memory copy streams.

| **PluggableDevice::Init** (SessionOption&amp; options) { se::Platform\* manager = PluggableDevManager();stream\_executor\_ = manager-\&gt; **ExecutorForDevice** ( **get\_id** (options));compute\_stream\_ = new se::Stream(stream\_executor\_);compute\_stream\_-\&gt; **Init** ();host\_to\_device\_stream\_ = new se::Stream(stream\_executor\_);host\_to\_device\_stream\_-\&gt; **Init** ();...} _// create StreamExecutor_ |
| --- |

3. PluggableDevicePlatform is responsible for the StreamExecutor creation. It creates an SE\_StreamExecutor and SE\_Device object through create\_stream\_executor and create\_device function handle which are registered in the SE\_Platform. Then PluggableDeviceExecutor is constructed with SE\_StreamExecutor and SE\_Device handle, which is an implementation instance of StreamExecutor.

| **PluggableDevicePlaform::ExeutorForDevice** (int device\_id） {auto config = get\_plugin\_config(device\_id);SE\_Options\* se\_option = get\_se\_option(device\_id);SE\_StreamExecutor\* se= platform\_-\&gt; **create\_stream\_executor** ();SE\_Device\* sd = platform\_-\&gt; **create\_device** (se\_options)auto executor = absl::make\_unique\&lt;StreamExecutor\&gt;(this, absl::make\_unique\&lt;PluggableDeviceExecutor\&gt;(config, se, sd)); **return** std::move(executor);} |
| --- |

**###Plugin**

Plugins need to implement and register the StreamExecutor C API defined in the Tensorflow proper.

- SE\_StreamExecutor is defined as struct in the C API, both sides(Tensorflow proper and plugins) can access its members. Plugin creates the SE\_StreamExecutor and registers its C API implementations to the SE\_StreamExecutor.

| SE\_StreamExecutor\* **create\_stream\_executor** () {SE\_StreamExecutor\* se\_nfs = new SE\_StreamExecutor();se-\&gt;memcpy\_from\_host = my\_device\_memory\_from\_host\_function;se-\&gt;allocate = my\_allocate\_function; …}_//Init device_ |
| --- |

- SE\_Device is defined as struct in the C API, both sides(Tensorflow proper and plugins) can access its members. Plugin creates the SE\_Device and fill its device opaque handle and device name to the SE\_Device.

| SE\_Device\* **create\_device** (SE\_Options\* options, TF\_Status\* status) {SE\_Device\* se = new SE\_Device();se-\&gt;device\_handle = get\_my\_device\_handle();...return se;} |
| --- |

- SE\_Stream is defined in plugin and treated as an opaque struct in Tensorflow proper.

| **void**** create\_stream**(SE\_Device\* executor, SE\_Stream\* stream, TF\_Status\*) {\*stream = new SE\_Stream\_st();(\*stream)-\&gt;stream\_handle = create\_my\_stream\_handle(executor);..} |
| --- |

MultiPlatformManager needs to be extended to identify the StreamExecutorPlatform associated with the pluggable device.

PluggableDeviceExecutor calls StreamExecutor C API to create the device?

### **Alternatives Considered**

- Make sure to discuss the relative merits of alternatives to your proposal.

### **Performance Implications**

- We don&#39;t expect performance impact due to this RFC. The functions described by this RFC are realized at the initialization stage.

### **Dependencies**

- This RFC doesn&#39;t add new dependencies

### **Engineering Impact**

- The impact to binary size / startup time / build time / test times are minimum.
- The TensorFlow team will maintain this code.

### **Platforms and Environments**

- Platforms: does this work on all platforms supported by TensorFlow? If not, why is that ok? Will it work on embedded/mobile? Does it impact automatic code generation or mobile stripping tooling? Will it work with transformation tools?
- Execution environments (Cloud services, accelerator hardware): what impact do you expect and how will you confirm?

### **Best Practices**

- Does this proposal change best practices for some aspect of using/developing TensorFlow? How will these changes be communicated/enforced?

### **Tutorials and Examples**

- If design changes existing API or creates new ones, the design owner should create end-to-end examples (ideally, a tutorial) which reflects how new feature will be used. Some things to consider related to the tutorial:
  - The minimum requirements for this are to consider how this would be used in a Keras-based workflow, as well as a non-Keras (low-level) workflow. If either isn&#39;t applicable, explain why.
    - It should show the usage of the new feature in an end to end example (from data reading to serving, if applicable). Many new features have unexpected effects in parts far away from the place of change that can be found by running through an end-to-end example. TFX [Examples](https://github.com/tensorflow/tfx/tree/master/tfx/examples) have historically been good in identifying such unexpected side-effects and are as such one recommended path for testing things end-to-end.
      - This should be written as if it is documentation of the new feature, i.e., consumable by a user, not a TensorFlow developer.
        - The code does not need to work (since the feature is not implemented yet) but the expectation is that the code does work before the feature can be merged.

### **Compatibility**

- Does the design conform to the backwards &amp; forwards compatibility [requirements](https://www.tensorflow.org/programmers_guide/version_compat)?
- How will this proposal interact with other parts of the TensorFlow Ecosystem?
  - How will it work with TFLite?
    - How will it work with distribution strategies?
      - How will it interact with tf.function?
        - Will this work on GPU/TPU?
          - How will it serialize to a SavedModel?

### **User Impact**

- What are the user-facing changes? How will this feature be rolled out?

## **Detailed Design**

This section is optional. Elaborate on details if they&#39;re important to understanding the design, but would make it hard to read the proposal section above.

## **Questions and Discussion Topics**

Seed this with open questions you require feedback on from the RFC process.
