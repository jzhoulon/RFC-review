<span class="c11 c34 c46">Pluggable device for Tensorflow</span> {#h.m0pos5xg5jqc .c36 .c9 .c55}
================================================================

[](){#t.c2e389f4528fdd572d039789113430b5b4585a2d}[](){#t.0}

+--------------------------------------+--------------------------------------+
| <span class="c4 c34">Status</span>   | <span class="c4 c34">(Proposed /     |
|                                      | Accepted / Implemented /             |
|                                      | Obsolete)</span>                     |
+--------------------------------------+--------------------------------------+
| <span class="c11 c4 c13">RFC         | <span                                |
| \#</span>                            | class="c52">[NNN](https://www.google |
|                                      | .com/url?q=https://github.com/tensor |
|                                      | flow/community/pull/NNN&sa=D&ust=159 |
|                                      | 2881106320000){.c5}</span><span      |
|                                      | class="c11 c4 c13"> (update when you |
|                                      | have community PR \#)</span>         |
+--------------------------------------+--------------------------------------+
| <span                                | <span class="c11 c4 c13">Zhoulong    |
| class="c11 c4 c13">Author(s)</span>  | Jiang, Yiqiang Li, Eric Lin, Jianhui |
|                                      | Li</span>                            |
+--------------------------------------+--------------------------------------+
| <span                                | <span class="c11 c4 c13">Anna        |
| class="c11 c4 c13">Sponsor</span>    | Revinskaya                           |
|                                      | (annarev@google.com)</span>          |
+--------------------------------------+--------------------------------------+
| <span                                | <span                                |
| class="c11 c4 c13">Updated</span>    | class="c11 c4 c13">2020-06-19</span> |
+--------------------------------------+--------------------------------------+

<span class="c11 c35 c34"></span> {#h.z6zy86s6wg0j .c36 .c9 .c37}
---------------------------------

<span class="c11 c35 c34">Objective</span> {#h.gxf0ujth6zo7 .c41 .c36 .c9}
------------------------------------------

<span class="c11 c4 c13">Implement a pluggable device mechanism which
allows to run existing tensorflow programs on a new device without user
changing the code.  Users only need to install a dynamic library in a
specified directory, and the mechanism is able to discover and plug in
the capabilities offered by the library. </span>

<span class="c4">This RFC is based on the Modular Tensorflow
 </span><span
class="c33 c47">[RFC](https://www.google.com/url?q=https://github.com/tensorflow/community/pull/77&sa=D&ust=1592881106323000){.c5}</span><span
class="c4">, which aims to extend the Tensorflow design to plugin
capabilities like adding a new device support.  The modular device
interface is based on StreamExecutor C API </span><span
class="c33 c47 c9">[RFC](https://www.google.com/url?q=https://github.com/tensorflow/community/pull/257&sa=D&ust=1592881106323000){.c5}</span><span
class="c11 c4 c13">. </span>

<span class="c11 c35 c34">Motivation</span> {#h.gii1g5racyaz .c41 .c36 .c9}
-------------------------------------------

<span class="c11 c4 c13">When extending Tensorflow to support a new
device, one needs to modify tensorflow code and maintain a special
tensorflow build for the new device. Modular Tensorflow RFC provides a
mechanism which adds the device support, built in a separate library, at
runtime.  This RFC further describes how tensorflow automatically
discovers these device libraries and adds them to tensorflow.  </span>

<span class="c11 c4 c13">The pluggable device discovery and
initialization is transparent to end users. As long as the device plugin
libraries follow the interface described in this RFC, it can be plugged
to tensorflow proper and enable tensorflow to run existing tensorflow
programs on a new device. </span>

<span class="c11 c34 c35">User Benefit</span> {#h.zad5ndy0m9eo .c41 .c36 .c9}
---------------------------------------------

<span class="c4">This allows tensorflow to transparently run tensorflow
programs on new devices, as long as users set up the system properly to
include device plugin libraries. </span>

<span class="c11 c35 c34">Design Proposal</span> {#h.k9jevhy9g33 .c36 .c9 .c41}
------------------------------------------------

<span class="c15 c11">Design Overview</span><span
style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 360.08px; height: 272.48px;">![](images/image1.png)</span>

<span class="c4">The RFC describes the mechanism of extending the
tensorflow device class hierarchy to add pluggable device as shown in
diagram 1. PluggableDevice is a virtual device defined in Tensorflow
proper which inherits LocalDevice.It is built on top of  StreamExecutor
C++ interface to manage PluggableDevice’s device, stream,  and memory.
 PluggableDeviceExecutor implements StreamExecutor and is built on top
of StreamExecutor C API (addressed in</span><span
class="c33 c47 c9">[ RFC](https://www.google.com/url?q=https://github.com/tensorflow/community/pull/257&sa=D&ust=1592881106324000){.c5}</span><span
class="c11 c4 c13">). </span>

<span class="c11 c4 c13"></span>

<span class="c11 c4 c13">PluggableDevice Backend is part of modular TF
plugin, which represents the physical device runtime. It implements
StreamExecutor C API and registers its platform to the Tensorflow proper
when the plugin’s shared object is loaded. </span>

<span class="c11 c4 c13"></span>

<span class="c11 c4 c13">The pluggable device mechanism contains device
discovery and creation process which creates a PluggableDevice object
and PluggableDeviceExecutor object for each PluggableDevice Backend.
</span>

<span class="c11 c4 c13">With the RFC, existing tensorflow GPU programs
can run on a plugged device without the user changing the code. The
diagram 2 describes the workflow of Tensorflow with device plugin, it
shows how a simple GPU program runs on the pluggable device.</span>

<span
style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 491.00px; height: 274.65px;">![](images/image2.png)</span>

<span class="c15 c11">Device Discovery</span>

<span class="c4">Upon initialization of Tensorflow, it uses platform
independent </span><span class="c4 c9">LoadLibrary() to load the dynamic
library. </span><span class="c4">The PluggableDevice Backend plugin
library should be installed to default plugin directory
“…python\_dir.../site-packages/tensorflow-plugins”. The modular
tensorflow </span><span
class="c33 c47">[RFC](https://www.google.com/url?q=https://github.com/tensorflow/community/pull/77&sa=D&ust=1592881106325000){.c5}</span><span
class="c4"> describes the process loading plugins. </span>

<span class="c0"></span>

<span class="c4 c9">During the plugin library initialization, it calls
the SE\_ReigsterPlatform() API to register the stream executor platform
(</span><span class="c4 c9">SE\_Platform </span><span class="c0">struct)
to Tensorflow proper. The SE\_ReigsterPlatform() API is a callback API,
part of StreamExecutor C API, which passes necessary information to
Tensorflow proper to instantiate a stream executor platform
(se::platform class) and register to a global object
MultiPlatformManager. </span>

<span class="c0"></span>

<span class="c0">The stream executor platform must be registered with
the name “PluggableDevice”.  </span>

<span class="c0"></span>

<span class="c0">See below code which is an example of registering a
PluggableDevice platform with StreamExecutor C API:</span>

<span class="c10 c11 c9">\`\`\`cpp</span>

<span class="c27 c9">void</span><span class="c10 c9"> </span><span
class="c20 c9">RegisterPluggableDevicePlatform</span><span
class="c10 c11 c9">() {</span>

<span class="c10 c11 c9">  static plugin\_id\_value = 123;</span>

<span class="c10 c11 c9">  SE\_PlatformId id;</span>

<span class="c10 c9">  id</span><span class="c9 c45">.id</span><span
class="c10 c11 c9"> = &plugin\_id\_value;</span>

<span class="c10 c11 c9">  int visible\_device\_count =
get\_plugin\_device\_count;</span>

<span class="c10 c11 c9"></span>

<span class="c10 c11 c9">  SE\_Platform\* custom\_platform =
SE\_NewPlatform(</span>

<span class="c10 c11 c9">     id, visible\_device\_count,</span>

<span class="c10 c11 c9">     create\_device,
create\_stream\_executor,</span>

<span class="c10 c11 c9">     delete\_device,
delete\_stream\_executor);</span>

<span class="c10 c11 c9"></span>

<span class="c10 c11 c9">  TF\_Status\* status = TF\_NewStatus();</span>

<span class="c10 c9">  std::string name = "</span><span
class="c10 c6">PluggableDevice</span><span class="c10 c11 c9">";</span>

<span class="c10 c9">  </span><span
class="c50 c9 c53">SE\_RegisterPlatform</span><span
class="c10 c11 c9">(</span>

<span class="c10 c9">     name.</span><span
class="c20 c9">c\_str</span><span class="c9 c10">(), name.</span><span
class="c20 c9">size</span><span class="c10 c11 c9">(),</span>

<span class="c10 c11 c9">     custom\_platform,</span>

<span class="c10 c11 c9">     status);</span>

<span class="c10 c11 c9">}</span>

<span class="c10 c11 c9">\`\`\`</span>

<span class="c10 c11 c9"></span>

<span class="c10 c9">Use </span><span class="c27 c9">static</span><span
class="c10 c9"> initialization to </span><span
class="c27 c9">register</span><span class="c10 c9"> the </span><span
class="c27 c9">new</span><span class="c10 c11 c9"> platform:</span>

<span class="c10 c11 c9"></span>

<span class="c10 c11 c9">\`\`\`cpp</span>

<span class="c27 c9">static</span><span class="c10 c9"> </span><span
class="c27 c9">bool</span><span
class="c10 c11 c9"> IsMyCustomPlatformRegistered = \[\]() {</span>

<span class="c10 c11 c9"> RegisterMyCustomPlatform();</span>

<span class="c10 c11 c9"> return true;</span>

<span class="c10 c11 c9">}();</span>

<span class="c10 c11 c9">\`\`\`</span>

<span class="c28 c9 c34">Device Creation</span>

<span class="c4 c9">PluggableDeviceFactory is introduced to create the
PluggableDevice, following the LocalDevice design pattern. To support
existing GPU programs run on a new device without user changing the code
, PluggableDeviceFactory is registered as “GPU” device name and given
higher priority than the default GPU. </span>

<span class="c6">   </span><span
class="c11 c13 c6 c39">REGISTER\_LOCAL\_DEVICE\_FACTORY("GPU",PluggableDeviceFactory,
220); // plugged GPU</span>

<span class="c11 c39 c13 c6">   REGISTER\_LOCAL\_DEVICE\_FACTORY("GPU",
GPUDeviceFactory, 210);//default GPU</span>

<span class="c11 c39 c13 c6"></span>

<span class="c0">When a session is created, PluggableDeviceFactory
creates a PluggableDevice object for the plugin device. During the
initialization of the PluggableDevice, a global object
MultiPlatformManager will find its se::platform through its platform
name: ”PluggableDevice”,  then stream executor platform (se::platform)
further creates a StreamExecutor object containing a
PluggableDeviceExecutor, and multiple stream objects(a computation
stream and several memory copy streams) supporting the StreamExecutor
objects. </span>

<span class="c0">See below the example code which creates the
PluggableDeviceExecutor using the information registered during plugin
library initialization. </span>

<span class="c0"></span>

<span class="c4 c9">The section below shows some pseudo code to
introduce some changes to the Tensorflow proper and what needs to be
implemented in the plugin for the pluggable device creation. The
implementation is based on </span><span class="c4">StreamExecutor C API
</span><span
class="c33 c47 c9">[RFC](https://www.google.com/url?q=https://github.com/tensorflow/community/pull/257&sa=D&ust=1592881106330000){.c5}</span><span
class="c4 c9">. </span>

<span class="c0"></span>

1.  <span class="c0">PluggableDeviceFactory creates and initializes a
    set of pluggable devices when the session is created.  </span>

<span class="c20 c9">PluggableDeviceFactory::CreateDevices</span><span
class="c10 c11 c9">(SessionOptions& options, const string& name\_prefix,
std::vector&lt;std::unique\_ptr&lt;Device&gt;&gt;\* devices) {</span>

<span class="c10 c9">  for (int i = 0; i &lt; options.</span><span
class="c20 c9">device\_count</span><span class="c10 c11 c9">(); i++)
{</span>

<span class="c10 c11 c9">    PluggableDevice pluggable\_device </span>

<span class="c10 c11 c9">    = CreatePluggableDevice(options,i); //set
allocator</span>

<span class="c10 c9">    pluggable\_device-&gt;</span><span
class="c20 c9">Init</span><span class="c10 c11 c9">(options);</span>

<span class="c10 c9">    devices.</span><span
class="c20 c9">push\_back</span><span class="c10 c9">(</span><span
class="c20 c9">std::move</span><span
class="c10 c11 c9">(pluggable\_device));</span>

<span class="c10 c11 c9">  }</span>

<span class="c10 c9">}</span>

2.  <span class="c0">PluggableDevice object binds a StreamExecutor and
    creates a set of Streams during the initialization.Streams include
    one compute stream and several memory copy streams.</span>

<span class="c7">PluggableDevice::Init</span><span
class="c14 c11 c9">(SessionOption& options) {  </span>

<span class="c14 c11 c9"> se::Platform\* platform=
se::MultiPlatformManager::</span>

<span class="c14 c11 c9">                       
 PlatformWithName(“PluggableDevice”);</span>

<span class="c30 c9"> stream\_executor\_ = platform-&gt;</span><span
class="c7">ExecutorForDevice</span><span
class="c14 c11 c9">(pluggable\_dev\_id\_);</span>

<span class="c14 c11 c9"> compute\_stream\_ = new
se::Stream(stream\_executor\_);</span>

<span class="c30 c9"> compute\_stream\_-&gt;</span><span
class="c7">Init</span><span class="c14 c11 c9">();</span>

<span class="c14 c11 c9"> host\_to\_device\_stream\_ = new
se::Stream(stream\_executor\_);</span>

<span class="c9 c30"> host\_to\_device\_stream\_-&gt;</span><span
class="c7">Init</span><span class="c14 c11 c9">();</span>

<span class="c11 c9 c14"> ...</span>

<span class="c30 c9">}  </span><span class="c30 c9 c49">// create
StreamExecutor</span>

3.  <span class="c0"> PluggableDevicePlatform is responsible for the
    StreamExecutor creation. It creates an SE\_StreamExecutor and
    SE\_Device object through create\_stream\_executor and
    create\_device function handle which are registered in
    the SE\_Platform. Then PluggableDeviceExecutor is then constructed
    with SE\_StreamExecutor and SE\_Device object.   </span>

<span
class="c20 c9">PluggableDevicePlaform::ExeutorForDevice</span><span
class="c10 c11 c9">(int device\_id） {</span>

<span class="c10 c11 c9">  auto config =
get\_plugin\_config(device\_id);</span>

<span class="c10 c11 c9">  SE\_Options\* se\_option =
get\_se\_option(device\_id);</span>

<span class="c10 c9">  SE\_StreamExecutor\* se=
platform\_-&gt;</span><span
class="c20 c9">create\_stream\_executor</span><span
class="c10 c11 c9">();</span>

<span class="c10 c9">  SE\_Device\* sd = platform\_-&gt;</span><span
class="c20 c9">create\_device</span><span
class="c10 c11 c9">(se\_options)</span>

<span class="c10 c9 c11">  auto executor =
absl::make\_unique&lt;StreamExecutor&gt;(this,
absl::make\_unique&lt;PluggableDeviceExecutor&gt;(config, se,
sd));</span>

<span class="c10 c9">  </span><span class="c27 c9">return</span><span
class="c10 c11 c9"> std::move(executor);</span>

<span class="c10 c9">}</span>

<span class="c11 c4 c9 c34 c48">\#\#\#Tensorflow Proper</span>

<span class="c0">Tensorflow proper needs to be extended to support a new
virtual device (PluggableDevice) to represent a set of new third-party
devices and a new stream executor platform (PluggableDevicePlatform) to
create the device and related resources with the information registered
from plugin. </span>

<span class="c0">Two sets of classes need to be defined in Tensorflow
proper. </span>

<span class="c0">Set 1: PluggableDevice related classes </span>

<span class="c4 c18">   class </span><span
class="c4 c18 c33">PluggableDevice</span><span class="c2"> : a virtual
device represents a set of new third-party devices, it has a new device
type named “PluggableDevice”/DEVICE\_PLUGGABLE. </span>

<span class="c4 c18">   class </span><span
class="c33 c4 c18">PluggableDeviceFactory</span><span class="c2">: a
device factory to create the PluggableDevice</span>

<span class="c4 c18">   class </span><span
class="c33 c4 c18">PluggableDeviceBFCAllocator</span><span class="c2">:
a PluggableDevice memory allocator that implements a ‘best fit with
coalescing’ algorithm.</span>

<span class="c4 c18">   class </span><span
class="c33 c4 c18">PluggableDeviceAllocato</span><span class="c2">r: an
allocator that wraps a PluggableDevice allocator.</span>

<span class="c4 c18">   class </span><span
class="c33 c4 c18">PluggableDeviceHostAllocator</span><span class="c2">:
allocator for pinned CPU RAM that is made known to PluggableDevice for
the purpose of efficient DMA with PluggableDevice.</span>

<span class="c4 c18">   class </span><span
class="c33 c4 c18">PluggableDeviceEventMgr</span><span class="c2">: an
object to keep track of pending Events in the StreamExecutor
streams.</span>

<span class="c4 c18">   class </span><span
class="c33 c4 c18">PluggableDeviceContext</span><span class="c2">: a
wrapper of pluggable device specific context that can be passed to
OpKernels.</span>

<span class="c0"></span>

<span class="c0">Set 2: PluggableDevicePlatform related classes </span>

<span class="c4 c6"> </span><span class="c4 c18">  class </span><span
class="c33 c4 c18">PluggableDevicePlatform</span><span class="c2"> :
PluggableDevice-specific platform, its platform name is
“PluggableDevice”, it contains a C struct: SE\_Platform\* platform\_
which is its internal implementation and as the C interface registered
by device plugin.</span>

<span class="c4 c18">   class</span><span
class="c33 c4 c18"> PluggableDeviceExecutor</span><span class="c2">:
PluggableDevice-platform implementation of the platform-agnostic
StreamExecutorInterface, it contains C structs: SE\_StreamExecutor\*
executor\_ and SE\_Device\* device\_ whose member can be accessed in
both Tensorflow proper and device plugins.</span>

<span class="c4 c6">  </span><span class="c4 c18"> class </span><span
class="c33 c4 c18">PluggableDeviceStream</span><span class="c2"> : wraps
a StreamHandle in order to satisfy the platform-independent
StreamInterface. It returns SE\_Stream which is treated as an opaque
type to Tensorflow,  whose structure is created by the device plugin.
 </span>

<span class="c4 c18">   class </span><span
class="c33 c4 c18">PluggableDeviceTimer</span><span class="c2"> : wraps
an opaque handle: SE\_Timer to satisfy the platform-independent
TimerInterface.</span>

<span class="c4 c18">   class </span><span
class="c33 c4 c18">PluggableDeviceEvent</span><span class="c2"> : wraps
an opaque handle: SE\_Event to satisfy the platform-independent
EventInterface.</span>

<span class="c2"></span>

<span class="c11 c4 c9 c34 c48">\#\#\#Plugin</span>

<span class="c0">Plugins need to implement and register the
StreamExecutor C API defined in the Tensorflow proper. </span>

-   <span class="c0">SE\_StreamExecutor is defined as struct in the C
    API, both sides(Tensorflow proper and plugins) can access
    its members. Plugin creates the SE\_StreamExecutor and registers its
    C API implementations to the SE\_StreamExecutor.  </span>

<span class="c10 c9">SE\_StreamExecutor\* </span><span
class="c20 c9">create\_stream\_executor</span><span
class="c10 c11 c9">() {</span>

<span class="c10 c11 c9">  SE\_StreamExecutor\* se\_nfs = new
SE\_StreamExecutor();</span>

<span class="c10 c11 c9">  se-&gt;memcpy\_from\_host =
my\_device\_memory\_from\_host\_function;</span>

<span class="c10 c11 c9">  se-&gt;allocate =
my\_allocate\_function;</span>

<span class="c10 c11 c9">  …</span>

<span class="c10 c9">}</span><span class="c49 c9 c50">//Init
device</span>

-   <span class="c0">SE\_Device is defined as struct in the C API, both
    sides(Tensorflow proper and plugins) can access its members. Plugin
    creates the SE\_Device and fills its device opaque handle and device
    name to the SE\_Device.</span>

<span class="c10 c9">SE\_Device\* </span><span
class="c9 c20">create\_device</span><span
class="c10 c11 c9">(SE\_Options\* options, TF\_Status\* status) {</span>

<span class="c10 c11 c9">  SE\_Device\* se = new SE\_Device();</span>

<span class="c10 c11 c9">  se-&gt;device\_handle =
get\_my\_device\_handle();</span>

<span class="c10 c11 c9">  ...</span>

<span class="c10 c11 c9">  return se;</span>

<span class="c10 c9">}</span>

-   <span class="c0">SE\_Stream is defined in plugin and treated as an
    opaque struct in Tensorflow proper. </span>

<span class="c27 c9">void</span><span class="c10 c9"> </span><span
class="c20 c9">create\_stream</span><span
class="c10 c11 c9">(SE\_Device\* executor, SE\_Stream\* stream,
TF\_Status\*) {</span>

<span class="c10 c11 c9">  \*stream = new SE\_Stream\_st();</span>

<span class="c10 c11 c9">  (\*stream)-&gt;stream\_handle =
create\_my\_stream\_handle(executor);</span>

<span class="c10 c11 c9">  ..</span>

<span class="c10 c9">}</span>

<span class="c15 c11 c9">\#\# PluggableDevice kernel registration
</span>

<span class="c4 c9">This RFC shows an example of registering kernels for
PluggableDevice. Kernel and op registration and implementation API is
addressed in a separate </span><span
class="c33 c47 c9">[RFC](https://www.google.com/url?q=https://github.com/tensorflow/community/blob/master/rfcs/20190814-kernel-and-op-registration.md&sa=D&ust=1592881106339000){.c5}</span><span
class="c0">. </span>

<span class="c0">Tensorflow proper defines a new device\_type named
DEVICE\_PLUGGABLE for PluggableDevice.This device\_type is used for the
kernel registration and dispatch. Plugin needs to register its kernel
implementation with DEVICE\_PLUGGABLE type.</span>

<span class="c27 c9">void</span><span class="c10 c9"> </span><span
class="c20 c9">InitPlugin</span><span class="c10 c11 c9">() {</span>

<span class="c10 c9">  TF\_KernelBuilder\* builder =
TF\_NewKernelBuilder(/\*op\_name\*/"Convolution", </span><span
class="c9 c34 c57">DEVICE\_PLUGGABLE</span><span
class="c10 c11 c9">,</span>

<span class="c10 c11 c9">      &Conv\_Create, &Conv\_Compute,
&Conv\_Delete);</span>

<span class="c10 c11 c9">  TF\_Status\* status = TF\_NewStatus();</span>

<span class="c10 c11 c9"> 
TF\_RegisterKernelBuilder(/\*kernel\_name\*/"Convolution", builder,
status);</span>

<span class="c10 c9">  if (</span><span
class="c20 c9">TF\_GetCode</span><span class="c10 c11 c9">(status) !=
TF\_OK) { /\* handle errors \*/ }</span>

<span class="c10 c11 c9">  TF\_DeleteStatus(status);</span>

<span class="c10 c9">}</span>

<span class="c28 c9 c34">\#\# Using stream inside PluggableDevice kernel
</span>

<span class="c0">The following code shows a convolution kernel
implementation using the stream handle. The streams are created during
the pluggable device creation. The placer decides which device to use
for each OP in the graph. Then the streams associated with the device
are used to construct the OpKernelContext for the op computation during
the graph execution.</span>

<span class="c9 c27">void</span><span class="c10 c9"> </span><span
class="c20 c9">Conv\_Compute</span><span
class="c10 c11 c9">(TF\_OpKernelContext\*) {</span>

<span class="c10 c11 c9">  TF\_GetInput(context, input\_index, &input,
&status);</span>

<span class="c10 c11 c9">  TF\_GetInput(context, filter\_index, &filter,
&status);</span>

<span class="c10 c11 c9">  auto output = TF\_AllocateOutput(context,
output\_index, TF\_Float32, dims, num\_dims, len, status);</span>

<span class="c10 c11 c9">  SE\_Stream se\_stream =
TF\_GetStream(TF\_OpKernelContext);</span>

<span class="c10 c11 c9">  auto native\_stream =
static\_cast&lt;native\_stream\_type&gt;(se\_stream-&gt;stream\_handle);</span>

<span class="c10 c11 c9">  my\_conv\_impl(input, filter, output,
native\_stream);</span>

<span class="c10 c9">}</span>

<span class="c4 c9">Kernel and op registration and implementation API
</span><span
class="c33 c47 c9">[RFC](https://www.google.com/url?q=https://github.com/tensorflow/community/blob/master/rfcs/20190814-kernel-and-op-registration.md&sa=D&ust=1592881106342000){.c5}</span><span
class="c0"> needs to be extended to retrieve streams/device context from
the TF\_OpKernelContext, besides inputs and outputs. </span>

### <span class="c12 c11">Alternatives Considered</span> {#h.2kias2wf34fl .c19 .c9}

-   <span class="c11 c4 c13">Without this RFC, end users need to change
    the python code to import the third-party device plugin. </span>
-   <span class="c4">Without this RFC, the third-party device vendor may
    implement the LocalDevice interface, which is not a C API interface
    and may interact with potential C++ ABI incompatibility issues.
     </span>

### <span class="c12 c11">Performance Implications</span> {#h.ifbq5be0h1wa .c9 .c19}

-   <span class="c4">We don’t expect performance impact due to this RFC.
    The functions described by this RFC are realized at the
    initialization stage. </span>

### <span class="c11 c12">Dependencies</span> {#h.cszsv8h2yp7o .c19 .c9}

-   <span class="c11 c4 c13">This RFC doesn’t add new dependencies to
    external libraries. </span>
-   <span class="c11 c4 c13">It depends on three modular Tensorflow
    related RFC </span>

<!-- -->

-   <span class="c4">Modular Tensorflow  </span><span
    class="c33 c47">[RFC](https://www.google.com/url?q=https://github.com/tensorflow/community/pull/77&sa=D&ust=1592881106343000){.c5}</span>
-   <span class="c4">StreamExecutor C</span><span class="c4"> interface
    </span><span
    class="c33 c9 c47">[RFC](https://www.google.com/url?q=https://github.com/tensorflow/community/pull/257&sa=D&ust=1592881106343000){.c5}</span>
-   <span class="c4 c9">Kernel and op registration and implementation
    API </span><span
    class="c33 c47 c9">[RFC](https://www.google.com/url?q=https://github.com/tensorflow/community/blob/master/rfcs/20190814-kernel-and-op-registration.md&sa=D&ust=1592881106343000){.c5}</span><span
    class="c0"> </span>

### <span class="c12 c11">Engineering Impact</span> {#h.affhswy9yen7 .c19 .c9}

-   <span class="c4">The impact to</span><span
    class="c11 c4 c13"> binary size / startup time / build time / test
    times are minimum. </span>
-   <span class="c11 c4 c13">The TensorFlow team will maintain
    this code. </span>

### <span class="c12 c11">Platforms and Environments</span> {#h.jcg97ye1x66a .c19 .c9}

-   <span class="c4">The pluggable device mechanism is based
    on loadlibrary() so should work on all the platforms supported
    by loadlibrary. The other enhancement to tensorflow proper is
    platform independent.</span>

### <span class="c12 c11">Best Practices</span> {#h.h0e46tqheq3h .c19 .c9}

-   <span class="c4">This works with Modular Tensorflow which will be
    the only way to integrate new third-party devices to the current
    Tensorflow stack. </span>

<span class="c11 c4 c13"></span>

### <span class="c12 c11">Compatibility</span> {#h.91t89bg7qmlh .c19 .c9}

<span class="c11 c4 c13">The RFC promotes the current Tensorflow
ecosystem as it supports plugging new devices to Tensorflow.  </span>

<span class="c4">We don't expect this proposal to impact </span><span
class="c11 c4 c13">with other parts of the Tensorflow ecosystem. It
doesn't support TFLite. It should not impede distribution strategies and
would not interact with tf.fuction and SaveModel.  </span>

<span class="c11 c17 c13"></span>
