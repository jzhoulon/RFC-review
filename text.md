<span class="c30 c64">Pluggable device for Tensorflow</span> {#h.m0pos5xg5jqc .c17 .c13 .c29}
============================================================

[](){#t.f0c0f2a242972cc9ae93d624e79945c00841f7aa}[](){#t.0}

+--------------------------------------+--------------------------------------+
| <span class="c9 c40">Status</span>   | <span class="c9 c40">(Proposed /     |
|                                      | Accepted / Implemented /             |
|                                      | Obsolete)</span>                     |
+--------------------------------------+--------------------------------------+
| <span class="c10 c9">RFC \#</span>   | <span                                |
|                                      | class="c50">[NNN](https://www.google |
|                                      | .com/url?q=https://github.com/tensor |
|                                      | flow/community/pull/NNN&sa=D&ust=159 |
|                                      | 2819032743000){.c45}</span><span     |
|                                      | class="c10 c9"> (update when you     |
|                                      | have community PR \#)</span>         |
+--------------------------------------+--------------------------------------+
| <span                                | <span class="c10 c9">Zhoulong Jiang, |
| class="c10 c9">Author(s)</span>      | Yiqiang Li, Eric Lin, Jianhui        |
|                                      | Li</span>                            |
+--------------------------------------+--------------------------------------+
| <span class="c10 c9">Sponsor</span>  | <span class="c10 c9">Anna Revinskaya |
|                                      | (annarev@google.com)</span>          |
+--------------------------------------+--------------------------------------+
| <span class="c10 c9">Updated</span>  | <span                                |
|                                      | class="c10 c9">2020-06-19</span>     |
+--------------------------------------+--------------------------------------+
| <span                                | <span class="c10 c9">TF-RFC it       |
| class="c9 c10">Obsoletes</span>      | replaces, else remove this           |
|                                      | header</span>                        |
+--------------------------------------+--------------------------------------+

<span class="c30 c35">Objective</span> {#h.z6zy86s6wg0j .c16 .c13}
--------------------------------------

<span class="c10 c9">Implement a pluggable device mechanism which allows
to run existing tensorflow programs on a new device without user
changing the code.  Users only need to install a dynamic library in a
specified directory, and the mechanism is able to discover and plug in
the capabilities offered by the library. </span>

<span class="c10 c9">This RFC is based on the Modular Tensorflow RFC,
which aims to extend the Tensorflow design to plugin capabilities like
adding a new device support.  The modular device interface is described
by a separate RFC. </span>

<span class="c30 c35">Motivation</span> {#h.gii1g5racyaz .c16 .c13}
---------------------------------------

<span class="c10 c9">When extending Tensorflow to support a new device,
one needs to modify tensorflow code and maintain a special tensorflow
build for the new device. Modular Tensorflow RFC provides a mechanism
which adds the device support, built in a separate library, at runtime.
 This RFC further describes how tensorflow automatically discovers these
device libraries and adds them to tensorflow.  </span>

<span class="c10 c9">The pluggable device discovery and initialization
is transparent to end users. As long as the device plugin libraries
follow the interface described in this RFC, it can be plugged to
tensorflow and run existing tensorflow programs targeting GPU device
type. </span>

<span class="c30 c35">User Benefit</span> {#h.zad5ndy0m9eo .c13 .c16}
-----------------------------------------

<span class="c9">This allows tensorflow to transparently run tensorflow
programs on new devices, as long as users set up the system properly to
include device plugin libraries. </span>

<span class="c30 c35">Design Proposal</span> {#h.k9jevhy9g33 .c16 .c13}
--------------------------------------------

<span class="c30 c46">Design Overview</span><span
style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 283.93px; height: 293.50px;">![](images/image1.png)</span>

<span class="c10 c9">The diagram 1. describes the mechanism of pluggable
device.</span>

<span class="c10 c9">PluggableDevice is a virtual device defined in
Tensorflow proper which inherits LocalDevice.It is built on top of
 StreamExecutor C++ interface to manage PluggableDevice’s device, stream
and data movement.</span>

<span class="c10 c9"></span>

<span class="c9">PluggableDeviceExecutor is StreamExecutor’s
implementation and built on top of StreamExecutor C API(addressed in
</span><span
class="c19 c13">[ RFC](https://www.google.com/url?q=https://github.com/tensorflow/community/pull/257&sa=D&ust=1592819032750000){.c45}</span><span
class="c9 c13"> </span><span class="c10 c9">). </span>

<span class="c10 c9"></span>

<span class="c10 c9">PluggableDevice Backend is part of modular TF
plugin, which represents the physical device runtime. It implements
 StreamExecutor C API and registers its platform to the Tensorflow
proper when the plugin’s shared object is loaded.  </span>

<span class="c10 c9"></span>

<span class="c10 c9">The pluggable device mechanism contains device
discovery and creation process which creates a PluggableDevice object
and PluggableDeviceExecutor object for each PluggableDevice Backend.
</span>

<span class="c10 c9">With this RFC, existing tensorflow GPU programs can
run on a plugged device without the user changing the code.The diagram 2
describes the workflow of Tensorflow with device plugin, it shows how a
simple GPU program runs on the pluggable device.</span>

<span
style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 491.00px; height: 274.65px;">![](images/image2.png)</span>

<span class="c30 c46">Device Discovery</span>

<span class="c9">The modular tensorflow </span><span
class="c19">[RFC](https://www.google.com/url?q=https://github.com/tensorflow/community/pull/77&sa=D&ust=1592819032752000){.c45}</span><span
class="c9"> describes the process loading plugins. The PluggableDevice
Backend plugin library should be installed to default plugin directory
“…python\_dir.../site-packages/tensorflow-plugins”. Upon initialization
of Tensorflow, it uses platform independent </span><span
class="c7">LoadLibrary() to load the dynamic library. </span>

<span class="c7"></span>

<span class="c9 c13">The plugin library implements the StreamExecutor C
API as defined in the</span><span
class="c19 c13">[ RFC](https://www.google.com/url?q=https://github.com/tensorflow/community/pull/257&sa=D&ust=1592819032753000){.c45}</span><span
class="c9 c13"> as well as t</span><span class="c7">he
SE\_ReigsterPlatform() API, which registers the platform and platform
name(PluggableDevice) to a global object named MultiPlatformManager in
Tensorflow proper during the library initialization. See below code
which is an example of registering a PluggableDevice platform with
StreamExecutor C API:</span>

[](){#t.a910eb2cadaef682320d332d6efd9f8224762866}[](){#t.1}

+--------------------------------------------------------------------------+
| <span class="c22 c13">\`\`\`cpp</span>                                   |
|                                                                          |
| <span class="c33 c13">void</span><span class="c23 c13"> </span><span     |
| class="c8">RegisterPluggableDevicePlatform</span><span                   |
| class="c22 c13">() {</span>                                              |
|                                                                          |
| <span class="c22 c13">  static plugin\_id\_value = 123;</span>           |
|                                                                          |
| <span class="c22 c13">  SE\_PlatformId id;</span>                        |
|                                                                          |
| <span class="c23 c13">  id</span><span class="c13 c60">.id</span><span   |
| class="c22 c13"> = &plugin\_id\_value;</span>                            |
|                                                                          |
| <span class="c22 c13">  int visible\_device\_count =                     |
| get\_plugin\_device\_count;</span>                                       |
|                                                                          |
| <span class="c22 c13"></span>                                            |
|                                                                          |
| <span class="c22 c13">  SE\_Platform\* custom\_platform =                |
| SE\_NewPlatform(</span>                                                  |
|                                                                          |
| <span class="c22 c13">     id, visible\_device\_count,</span>            |
|                                                                          |
| <span class="c22 c13">     create\_device,                               |
| create\_stream\_executor,</span>                                         |
|                                                                          |
| <span class="c22 c13">     delete\_device,                               |
| delete\_stream\_executor);</span>                                        |
|                                                                          |
| <span class="c22 c13"></span>                                            |
|                                                                          |
| <span class="c22 c13">  TF\_Status\* status = TF\_NewStatus();</span>    |
|                                                                          |
| <span class="c23 c13">  std::string name = "</span><span                 |
| class="c23 c39">PluggableDevice</span><span class="c22 c13">";</span>    |
|                                                                          |
| <span class="c22 c13">  SE\_RegisterPlatform(</span>                     |
|                                                                          |
| <span class="c23 c13">     name.</span><span                             |
| class="c8">c\_str</span><span class="c23 c13">(), name.</span><span      |
| class="c8">size</span><span class="c22 c13">(),</span>                   |
|                                                                          |
| <span class="c22 c13">     custom\_platform,</span>                      |
|                                                                          |
| <span class="c22 c13">     status);</span>                               |
|                                                                          |
| <span class="c22 c13">}</span>                                           |
|                                                                          |
| <span class="c22 c13">\`\`\`</span>                                      |
|                                                                          |
| <span class="c22 c13"></span>                                            |
|                                                                          |
| <span class="c23 c13">Use </span><span                                   |
| class="c33 c13">static</span><span class="c23 c13"> initialization to    |
| </span><span class="c13 c33">register</span><span class="c23 c13"> the   |
| </span><span class="c33 c13">new</span><span                             |
| class="c22 c13"> platform:</span>                                        |
|                                                                          |
| <span class="c22 c13"></span>                                            |
|                                                                          |
| <span class="c22 c13">\`\`\`cpp</span>                                   |
|                                                                          |
| <span class="c33 c13">static</span><span class="c23 c13"> </span><span   |
| class="c33 c13">bool</span><span                                         |
| class="c22 c13"> IsMyCustomPlatformRegistered = \[\]() {</span>          |
|                                                                          |
| <span class="c22 c13"> RegisterMyCustomPlatform();</span>                |
|                                                                          |
| <span class="c22 c13"> return true;</span>                               |
|                                                                          |
| <span class="c22 c13">}();</span>                                        |
+--------------------------------------------------------------------------+

<span class="c46 c40 c13">Device Creation</span>

<span class="c9 c13">PluggableDeviceFactory is introduced to create the
PluggableDevice, following the LocalDevice design pattern. To support
existing GPU programs run on a new device without user changing the code
, PluggableDeviceFactory is registered as “GPU” device name and given
higher priority than the default GPU. </span>

<span class="c39">   </span><span
class="c10 c63 c39">REGISTER\_LOCAL\_DEVICE\_FACTORY("GPU",PluggableDeviceFactory,
220); // plugged GPU</span>

<span class="c10 c63 c39">   REGISTER\_LOCAL\_DEVICE\_FACTORY("GPU",
GPUDeviceFactory, 210);//default GPU</span>

<span class="c10 c39 c63"></span>

<span class="c9 c13">When a session is created, PluggableDeviceFactory
creates a PluggableDevice object for the plugin device. During the
initialization of the PluggableDevice, a global object
MultiPlatformManager will find its se::platform through its platform
name: ”PluggableDevice”,  then </span><span
class="c9 c13">StreamExecutorPlatform
</span>^[\[a\]](#cmnt1){#cmnt_ref1}[\[b\]](#cmnt2){#cmnt_ref2}^<span
class="c7">(se::platform) further creates a StreamExecutor object
containing a PluggableDeviceExecutor, and multiple stream objects(a
computation stream and several memory copy streams) supporting the
StreamExecutor objects. </span>

<span class="c30 c13 c43">\#\#Implementation</span>

<span class="c9 c13">This section shows some pseudo code to introduce
some changes to the Tensorflow proper and what needs to be implemented
in the plugin for the pluggable device creation. The implementation is
based on </span><span class="c13 c19">[StreamExecutor C API
RFC](https://www.google.com/url?q=https://github.com/tensorflow/community/pull/257&sa=D&ust=1592819032762000){.c45}</span>

<span class="c30 c9 c13">\#\#\#Tensorflow Proper</span>

<span class="c7">Tensorflow proper will add a new virtual device named
PluggableDevice which represents a set of new third-party
devices.Following the LocalDevice design, a set of class need to be
defined in Tensorflow proper:</span>

<span class="c9 c15">   class </span><span
class="c26 c9 c15">PluggableDevice</span><span class="c10 c9 c15"> : a
virtual device represents a set of new third-party devices, it has a new
device type named “PluggableDevice”/DEVICE\_PLUGGABLE. </span>

<span class="c9 c15">   class </span><span
class="c26 c9 c15">PluggableDeviceFactory</span><span
class="c10 c9 c15">: a device factory to create the
PluggableDevice</span>

<span class="c9 c15">   class </span><span
class="c26 c9 c15">PluggableDeviceBFCAllocator</span><span
class="c10 c9 c15">: a PluggableDevice memory allocator that implements
a ‘best fit with coalescing’ algorithm.</span>

<span class="c9 c15">   class </span><span
class="c26 c9 c15">PluggableDeviceAllocato</span><span
class="c10 c9 c15">r: an allocator that wraps a PluggableDevice
allocator.</span>

<span class="c9 c15">   class </span><span
class="c26 c9 c15">PluggableDeviceHostAllocator</span><span
class="c10 c9 c15">: allocator for pinned CPU RAM that is made known to
PluggableDevice for the purpose of efficient DMA with
PluggableDevice.</span>

<span class="c9 c15">   class </span><span
class="c26 c9 c15">PluggableDeviceEventMgr</span><span
class="c10 c9 c15">: an object to keep track of pending Events in the
StreamExecutor streams.</span>

<span class="c9 c15">   class </span><span
class="c26 c9 c15">PluggableDeviceContext</span><span
class="c10 c9 c15">: a wrapper of pluggable device specific context that
can be passed to OpKernels.</span>

<span class="c7">Tensorflow proper will add a new StreamExecutor
Platform named PluggableDevicePlatform whose implementation is
registered in plugin.</span>

<span class="c9 c39"> </span><span class="c9 c15">  class </span><span
class="c26 c9 c15">PluggableDevicePlatform</span><span
class="c10 c9 c15"> : PluggableDevice-specific platform, its platform
name is “PluggableDevice”, it contains a C struct: SE\_Platform\*
platform\_ which is its internal implementation and as the C interface
registered by device plugin.</span>

<span class="c9 c15">   class</span><span
class="c26 c9 c15"> PluggableDeviceExecutor</span><span
class="c10 c9 c15">: PluggableDevice-platform implementation of the
platform-agnostic StreamExecutorInterface, it contains C structs:
SE\_StreamExecutor\* executor\_ and SE\_Device\* device\_ whose member
can be accessed in both Tensorflow proper and device plugins.</span>

<span class="c9 c39">  </span><span class="c9 c15"> class </span><span
class="c26 c9 c15">PluggableDeviceStream</span><span
class="c10 c9 c15"> : wraps a StreamHandle in order to satisfy the
platform-independent StreamInterface. It returns SE\_Stream which is
treated as an opaque type to Tensorflow,  whose structure is created by
the device plugin.  </span>

<span class="c9 c15">   class </span><span
class="c9 c15 c26">PluggableDeviceTimer</span><span
class="c10 c9 c15"> : wraps an opaque handle: SE\_Timer to satisfy the
platform-independent TimerInterface.</span>

<span class="c9 c15">   class </span><span
class="c26 c9 c15">PluggableDeviceEvent</span><span
class="c10 c9 c15"> : wraps an opaque handle: SE\_Event to satisfy the
platform-independent EventInterface.</span>

<span class="c7">The following pseudocode shows the process of
PluggableDevice creation.</span>

1.  <span class="c7">PluggableDeviceFactory creates and initializes a
    set of pluggable devices when the session is created.  </span>

[](){#t.4f0ef52f9144fdfc9ab41524137a4f01c5483cdf}[](){#t.2}

+--------------------------------------------------------------------------+
| <span class="c8">PluggableDeviceFactory::CreateDevices</span><span       |
| class="c22 c13">(SessionOptions& options, const string& name\_prefix,    |
| std::vector&lt;std::unique\_ptr&lt;Device&gt;&gt;\* devices) {</span>    |
|                                                                          |
| <span class="c23 c13">  for (int i = 0; i &lt; options.</span><span      |
| class="c8">device\_count</span><span class="c22 c13">(); i++) {</span>   |
|                                                                          |
| <span class="c22 c13">    PluggableDevice pluggable\_device </span>      |
|                                                                          |
| <span class="c22 c13">    = CreatePluggableDevice(options,i); //set      |
| allocator</span>                                                         |
|                                                                          |
| <span class="c23 c13">    pluggable\_device-&gt;</span><span             |
| class="c8">Init</span><span class="c22 c13">(options);</span>            |
|                                                                          |
| <span class="c23 c13">    devices.</span><span                           |
| class="c8">push\_back</span><span class="c23 c13">(</span><span          |
| class="c8">std::move</span><span                                         |
| class="c22 c13">(pluggable\_device));</span>                             |
|                                                                          |
| <span class="c22 c13">  }</span>                                         |
|                                                                          |
| <span class="c23 c13">}</span>                                           |
+--------------------------------------------------------------------------+

2.  <span class="c7">PluggableDevice object will bind to a
    StreamExecutor and creates a set of Streams during the
    initialization.Streams include one compute stream and several memory
    copy streams.</span>

[](){#t.23af4ea2e3d8ba897b7e803472b76dd938b06526}[](){#t.3}

+--------------------------------------------------------------------------+
| <span class="c27 c13">PluggableDevice::Init</span><span                  |
| class="c14 c13">(SessionOption& options) {  </span>                      |
|                                                                          |
| <span class="c13 c14"> se::Platform\* platform=                          |
| se::MultiPlatformManager::</span>                                        |
|                                                                          |
| <span class="c14 c13">                                                   |
|  PlatformWithName(“PluggableDevice”);</span>                             |
|                                                                          |
| <span class="c34 c13"> stream\_executor\_ = platform-&gt;</span><span    |
| class="c27 c13">ExecutorForDevice</span><span                            |
| class="c34 c13">(</span><span                                            |
| class="c34 c13">pluggable\_dev\_id\_</span><span                         |
| class="c14 c13">);</span>                                                |
|                                                                          |
| <span class="c14 c13"> compute\_stream\_ = new                           |
| se::Stream(stream\_executor\_);</span>                                   |
|                                                                          |
| <span class="c34 c13"> compute\_stream\_-&gt;</span><span                |
| class="c27 c13">Init</span><span class="c14 c13">();</span>              |
|                                                                          |
| <span class="c14 c13"> host\_to\_device\_stream\_ = new                  |
| se::Stream(stream\_executor\_);</span>                                   |
|                                                                          |
| <span class="c34 c13"> host\_to\_device\_stream\_-&gt;</span><span       |
| class="c13 c27">Init</span><span class="c14 c13">();</span>              |
|                                                                          |
| <span class="c14 c13"> ...</span>                                        |
|                                                                          |
| <span class="c13 c34">}  </span><span class="c34 c13 c56">// create      |
| StreamExecutor</span>                                                    |
+--------------------------------------------------------------------------+

3.  <span class="c7"> PluggableDevicePlatform is responsible for the
    StreamExecutor creation. It creates an SE\_StreamExecutor and
    SE\_Device object through create\_stream\_executor and
    create\_device function handle which are registered in
    the SE\_Platform. Then PluggableDeviceExecutor is constructed with
    SE\_StreamExecutor and SE\_Device handle, which is an implementation
    instance of StreamExecutor.  </span>

[](){#t.3b6ca2cb5c676383b4ec6445d20b8b4c6545f4e5}[](){#t.4}

+--------------------------------------------------------------------------+
| <span class="c8">PluggableDevicePlaform::ExeutorForDevice</span><span    |
| class="c22 c13">(int device\_id） {</span>                               |
|                                                                          |
| <span class="c22 c13">  auto config =                                    |
| get\_plugin\_config(device\_id);</span>                                  |
|                                                                          |
| <span class="c22 c13">  SE\_Options\* se\_option =                       |
| get\_se\_option(device\_id);</span>                                      |
|                                                                          |
| <span class="c23 c13">  SE\_StreamExecutor\* se=                         |
| platform\_-&gt;</span><span                                              |
| class="c8">create\_stream\_executor</span><span                          |
| class="c22 c13">();</span>                                               |
|                                                                          |
| <span class="c23 c13">  SE\_Device\* sd = platform\_-&gt;</span><span    |
| class="c8">create\_device</span><span                                    |
| class="c22 c13">(se\_options)</span>                                     |
|                                                                          |
| <span class="c22 c13">  auto executor =                                  |
| absl::make\_unique&lt;StreamExecutor&gt;(this,                           |
| absl::make\_unique&lt;PluggableDeviceExecutor&gt;(config, se,            |
| sd));</span>                                                             |
|                                                                          |
| <span class="c23 c13">  </span><span class="c33 c13">return</span><span  |
| class="c22 c13"> std::move(executor);</span>                             |
|                                                                          |
| <span class="c23 c13">}</span>                                           |
+--------------------------------------------------------------------------+

<span class="c9 c13 c30">\#\#\#Plugin</span>

<span class="c7">Plugins need to implement and register the
StreamExecutor C API defined in the Tensorflow proper. </span>

-   <span class="c7">SE\_StreamExecutor is defined as struct in the C
    API, both sides(Tensorflow proper and plugins) can access
    its members. Plugin creates the SE\_StreamExecutor and registers its
    C API implementations to the SE\_StreamExecutor.  </span>

[](){#t.7d0018e0fb21eb218296865404520331fb8b5ef2}[](){#t.5}

+--------------------------------------------------------------------------+
| <span class="c23 c13">SE\_StreamExecutor\* </span><span                  |
| class="c8">create\_stream\_executor</span><span class="c22 c13">()       |
| {</span>                                                                 |
|                                                                          |
| <span class="c22 c13">  SE\_StreamExecutor\* se\_nfs = new               |
| SE\_StreamExecutor();</span>                                             |
|                                                                          |
| <span class="c22 c13">  se-&gt;memcpy\_from\_host =                      |
| my\_device\_memory\_from\_host\_function;</span>                         |
|                                                                          |
| <span class="c13 c22">  se-&gt;allocate = my\_allocate\_function;</span> |
|                                                                          |
| <span class="c22 c13">  …</span>                                         |
|                                                                          |
| <span class="c23 c13">}</span><span class="c13 c53">//Init device</span> |
+--------------------------------------------------------------------------+

-   <span class="c7">SE\_Device is defined as struct in the C API, both
    sides(Tensorflow proper and plugins) can access its members. Plugin
    creates the SE\_Device and fill its device opaque handle and device
    name to the SE\_Device.</span>

[](){#t.f6ee87282e63e10d6e68bab63f5b66c501a6d6ca}[](){#t.6}

+--------------------------------------------------------------------------+
| <span class="c23 c13">SE\_Device\* </span><span                          |
| class="c8">create\_device</span><span class="c22 c13">(SE\_Options\*     |
| options, TF\_Status\* status) {</span>                                   |
|                                                                          |
| <span class="c22 c13">  SE\_Device\* se = new SE\_Device();</span>       |
|                                                                          |
| <span class="c22 c13">  se-&gt;device\_handle =                          |
| get\_my\_device\_handle();</span>                                        |
|                                                                          |
| <span class="c22 c13">  ...</span>                                       |
|                                                                          |
| <span class="c22 c13">  return se;</span>                                |
|                                                                          |
| <span class="c23 c13">}</span>                                           |
+--------------------------------------------------------------------------+

-   <span class="c7">SE\_Stream is defined in plugin and treated as an
    opaque struct in Tensorflow proper. </span>

[](){#t.7186ada6f6f511da2feb1e1d23d453c3b04bdc85}[](){#t.7}

+--------------------------------------------------------------------------+
| <span class="c33 c13">void</span><span class="c23 c13"> </span><span     |
| class="c8">create\_stream</span><span class="c22 c13">(SE\_Device\*      |
| executor, SE\_Stream\* stream, TF\_Status\*) {</span>                    |
|                                                                          |
| <span class="c22 c13">  \*stream = new SE\_Stream\_st();</span>          |
|                                                                          |
| <span class="c22 c13">  (\*stream)-&gt;stream\_handle =                  |
| create\_my\_stream\_handle(executor);</span>                             |
|                                                                          |
| <span class="c22 c13">  ..</span>                                        |
|                                                                          |
| <span class="c13 c23">}</span>                                           |
+--------------------------------------------------------------------------+

<span class="c46 c40 c13">Kernel Registration and Implementation</span>

<span class="c9 c13">Kernel and op registration and implementation API
is addressed in another </span><span
class="c19 c13">[RFC](https://www.google.com/url?q=https://github.com/tensorflow/community/blob/master/rfcs/20190814-kernel-and-op-registration.md&sa=D&ust=1592819032781000){.c45}</span><span
class="c7">, this RFC will show some examples of using those C API to
implement kernels for PluggableDevice.</span>

<span class="c7">\#\#Kernel Registration</span>

<span class="c7"> Tensorflow proper will add a new device\_type named
DEVICE\_PLUGGABLE for PluggableDevice.This device\_type will be used for
the kernel registration and dispatch, and some runtime divergence with
other device types(DEVICE\_GPU, DEVICE\_CPU..).</span>

<span class="c7">Plugin needs to register its kernel implementation with
DEVICE\_PLUGGABLE type.</span>

[](){#t.09f9a48df317c11be20412d7e379b775ad6df15d}[](){#t.8}

+--------------------------------------------------------------------------+
| <span class="c33 c13">void</span><span class="c23 c13"> </span><span     |
| class="c8">InitPlugin</span><span class="c22 c13">() {</span>            |
|                                                                          |
| <span class="c23 c13">  TF\_KernelBuilder\* builder =                    |
| TF\_NewKernelBuilder(/\*op\_name\*/"Convolution", </span><span           |
| class="c40 c13 c65">DEVICE\_PLUGGABLE</span><span                        |
| class="c22 c13">,</span>                                                 |
|                                                                          |
| <span class="c22 c13">      &Conv\_Create, &Conv\_Compute,               |
| &Conv\_Delete);</span>                                                   |
|                                                                          |
| <span class="c22 c13">  TF\_Status\* status = TF\_NewStatus();</span>    |
|                                                                          |
| <span class="c22 c13">                                                   |
| TF\_RegisterKernelBuilder(/\*kernel\_name\*/"Convolution", builder,      |
| status);</span>                                                          |
|                                                                          |
| <span class="c23 c13">  if (</span><span                                 |
| class="c8">TF\_GetCode</span><span class="c22 c13">(status) != TF\_OK) { |
| /\* handle errors \*/ }</span>                                           |
|                                                                          |
| <span class="c22 c13">  TF\_DeleteStatus(status);</span>                 |
|                                                                          |
| <span class="c23 c13">}</span>                                           |
+--------------------------------------------------------------------------+

<span class="c7">\#\#Kernel Implementation</span>

<span class="c7">Kernel and Op Implementation and Registration API RFC
has defined APIs for retrieving inputs and outputs from the
TF\_OpKernelContext, this RFC assumes it will also has the API for
retrieving streams/device context from the TF\_OpKernelContext. The
following code is to show an simple example of convolution
compute:</span>

[](){#t.dbd8fb83ba25135c515523e9852efba1c453c4a7}[](){#t.9}

+--------------------------------------------------------------------------+
| <span class="c33 c13">void</span><span class="c23 c13"> </span><span     |
| class="c8">Conv\_Compute</span><span                                     |
| class="c22 c13">(TF\_OpKernelContext\*) {</span>                         |
|                                                                          |
| <span class="c22 c13">  TF\_GetInput(context, input\_index, &input,      |
| &status);</span>                                                         |
|                                                                          |
| <span class="c22 c13">  TF\_GetInput(context, filter\_index, &filter,    |
| &status);</span>                                                         |
|                                                                          |
| <span class="c22 c13">  auto output = TF\_AllocateOutput(context,        |
| output\_index, TF\_Float32, dims, num\_dims, len, status);</span>        |
|                                                                          |
| <span class="c22 c13">  SE\_Stream se\_stream =                          |
| TF\_GetStream(TF\_OpKernelContext);</span>                               |
|                                                                          |
| <span class="c22 c13">  auto native\_stream =                            |
| static\_cast&lt;native\_stream\_type&gt;(se\_stream-&gt;stream\_handle); |
| </span>                                                                  |
|                                                                          |
| <span class="c22 c13">  my\_conv\_impl(input, filter, output,            |
| native\_stream);</span>                                                  |
|                                                                          |
| <span class="c23 c13">}</span>                                           |
+--------------------------------------------------------------------------+

<span class="c7"></span>

<span class="c7">MultiPlatformManager needs to be extended to identify
the StreamExecutorPlatform associated with the pluggable device.
 </span>

<span class="c9 c13">PluggableDeviceExecutor calls StreamExecutor C API
to create the device? </span>

<span class="c10 c9"></span>

### <span class="c30 c49">Alternatives Considered</span> {#h.2kias2wf34fl .c24 .c17 .c13}

-   <span class="c10 c9">Make sure to discuss the relative merits of
    alternatives to your proposal.</span>

### <span class="c30 c49">Performance Implications</span> {#h.ifbq5be0h1wa .c24 .c17 .c13}

-   <span class="c9">We don’t expect performance impact due to this RFC.
    The functions described by this RFC are realized at the
    initialization stage. </span>

### <span class="c30 c49">Dependencies</span> {#h.cszsv8h2yp7o .c24 .c17 .c13}

-   <span class="c9">This RFC doesn’t add new dependencies</span>

### <span class="c30 c49">Engineering Impact</span> {#h.affhswy9yen7 .c24 .c17 .c13}

-   <span class="c9">The impact to</span><span class="c10 c9"> binary
    size / startup time / build time / test times are minimum. </span>
-   <span class="c10 c9">The TensorFlow team will maintain this code.
    </span>

### <span class="c30 c49">Platforms and Environments</span> {#h.jcg97ye1x66a .c24 .c17 .c13}

-   <span class="c10 c9">Platforms: does this work on all platforms
    supported by TensorFlow? If not, why is that ok? Will it work on
    embedded/mobile? Does it impact automatic code generation or mobile
    stripping tooling? Will it work with transformation tools?</span>
-   <span class="c10 c9">Execution environments (Cloud services,
    accelerator hardware): what impact do you expect and how will you
    confirm?</span>

### <span class="c30 c49">Best Practices</span> {#h.h0e46tqheq3h .c24 .c17 .c13}

-   <span class="c10 c9">Does this proposal change best practices for
    some aspect of using/developing TensorFlow? How will these changes
    be communicated/enforced?</span>

### <span class="c30 c49">Tutorials and Examples</span> {#h.ub36sh9kvktc .c24 .c17 .c13}

-   <span class="c10 c9">If design changes existing API or creates new
    ones, the design owner should create end-to-end examples (ideally,
    a tutorial) which reflects how new feature will be used. Some things
    to consider related to the tutorial:</span>

<!-- -->

-   <span class="c10 c9">The minimum requirements for this are to
    consider how this would be used in a Keras-based workflow, as well
    as a non-Keras (low-level) workflow. If either isn’t applicable,
    explain why.</span>
-   <span class="c9">It should show the usage of the new feature in an
    end to end example (from data reading to serving, if applicable).
    Many new features have unexpected effects in parts far away from the
    place of change that can be found by running through an
    end-to-end example. TFX </span><span
    class="c50">[Examples](https://www.google.com/url?q=https://github.com/tensorflow/tfx/tree/master/tfx/examples&sa=D&ust=1592819032790000){.c45}</span><span
    class="c10 c9"> have historically been good in identifying such
    unexpected side-effects and are as such one recommended path for
    testing things end-to-end.</span>
-   <span class="c10 c9">This should be written as if it is
    documentation of the new feature, i.e., consumable by a user, not a
    TensorFlow developer.</span>
-   <span class="c10 c9">The code does not need to work (since the
    feature is not implemented yet) but the expectation is that the code
    does work before the feature can be merged.</span>

### <span class="c30 c49">Compatibility</span> {#h.91t89bg7qmlh .c17 .c13 .c24}

-   <span class="c9">Does the design conform to the backwards & forwards
    compatibility </span><span
    class="c50">[requirements](https://www.google.com/url?q=https://www.tensorflow.org/programmers_guide/version_compat&sa=D&ust=1592819032791000){.c45}</span><span
    class="c10 c9">?</span>
-   <span class="c10 c9">How will this proposal interact with other
    parts of the TensorFlow Ecosystem?</span>

<!-- -->

-   <span class="c10 c9">How will it work with TFLite?</span>
-   <span class="c10 c9">How will it work with distribution
    strategies?</span>
-   <span class="c10 c9">How will it interact with tf.function?</span>
-   <span class="c10 c9">Will this work on GPU/TPU?</span>
-   <span class="c10 c9">How will it serialize to a SavedModel?</span>

### <span class="c30 c49">User Impact</span> {#h.uo2pl4o5u2z3 .c24 .c17 .c13}

-   <span class="c10 c9">What are the user-facing changes? How will this
    feature be rolled out?</span>

<span class="c30 c35">Detailed Design</span> {#h.h53i7qnu8itr .c16 .c13}
--------------------------------------------

<span class="c10 c9">This section is optional. Elaborate on details if
they’re important to understanding the design, but would make it hard to
read the proposal section above.</span>

<span class="c30 c35">Questions and Discussion Topics</span> {#h.nzo5igpgmj2l .c16 .c13}
------------------------------------------------------------

<span class="c10 c9">Seed this with open questions you require feedback
on from the RFC process.</span>

<span class="c10 c20"></span>

<div class="c66">

[\[a\]](#cmnt_ref1){#cmnt1}<span class="c10 c20">@Zhoulong, how does
this link to "MultiPlatformManager" in previous paragraph?</span>

</div>

<div class="c66">

[\[b\]](#cmnt_ref2){#cmnt2}<span class="c10 c20">platform was found by
MutiPlatformManager through its name</span>

</div>
