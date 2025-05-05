.. highlight:: python

使用可执行的特征提取器
======================

简介
----

虽然 Essentia 主要作为库使用，但它也包含了许多示例代码，这些代码可以在主代码库之外的 ``src/examples`` 文件夹中找到。其中一些是可执行文件，可以用来计算给定音频文件的音乐信息检索（MIR）描述符。编译后，这些可执行文件将位于 ``build/src/examples`` 文件夹中。

特征提取器
----------

这些示例包含几个可执行的特征提取器，你可以使用它们来：
- 熟悉 Essentia 能够计算的各种描述符类型
- 在构建自己的提取器时作为参考

可用的提取器包括：

* ``streaming_extractor_music``：计算大量频谱、时域、节奏、音调和高级描述符。帧级描述符通过其统计分布进行汇总。该提取器专为大型音乐集合的批量计算而设计。详见 `详细文档 <streaming_extractor_music.html>`_。

* ``streaming_extractor_freesound``：类似的特征提取器，推荐用于声音分析。该提取器被 `Freesound <http://freesound.org>`_ 用于提供声音分析 API 和相似声音搜索功能。

* ``streaming_extractor``：过时的提取器，包含大量描述符和分段功能。其描述符集包含一些不稳定的描述符，可靠性低于 ``streaming_extractor_music``。

* ``standard_pitchyinfft``：使用 `YinFFT <reference/std_PitchYinFFT.html>`_ 算法提取单音信号的音高。

* ``streaming_predominantmelody``：使用 `MELODIA <reference/std_PredominantMelody.html>`_ 算法提取主旋律的音高。

* ``streaming_beattracker_multifeature_mirex2013``：使用 `多特征节拍跟踪器 <reference/std_BeatTrackerMultiFeature.html>`_ 算法提取节拍位置。

* ``streaming_mfcc``：提取 MFCC（梅尔频率倒谱系数）帧及其统计特征。

* ``standard_rhythmtransform``：计算 `节奏变换 <reference/std_RhythmTransform.html>`_。

给定一个音频文件，这些提取器会生成一个包含结果的 yaml 或 json 文件。 