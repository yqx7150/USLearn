# Code for "MR image reconstruction using deep density priors", IEEE TMI, 2018
## K.Tezcan, C. Baumgartner, R. Luechinger, K. Pruessmann, E. Konukoglu
### tezcan@vision.ee.ethz.ch, CVL ETH Zürich
#### Link to paper: https://ieeexplore.ieee.org/document/8579232
Supplementary materials:
https://ieeexplore.ieee.org/ielx7/42/4359023/8579232/multimedia.pdf?tp=&arnumber=8579232

Please do not hesitate to contact for questions or feedback.

Files:

1. runrecon.py: The file that prepares and runs the reconstruction for the sample image from the HCP dataset. This loads the image, creates the undersampled version and calls the reconstruction function: vaerecon.py. After the reconstruction the reconstructed and the zero-filled images are saved as pickle files: 'rec' and 'zerofilled'. 'rec' contains the images throughout the iterations, so can be used to calculate step-wise RMSE to verify convergence.
runrecon.py：从HCP数据集准备并运行样本图像重建的文件。 这会加载图像，创建欠采样版本并调用重建函数：vaerecon.py。 在重建之后，重建的和零填充的图像被保存为pickle文件：'rec'和'zerofilled'。 'rec'包含整个迭代中的图像，因此可用于计算逐步RMSE以验证收敛。

2. runrecon_acquired_image.py: The file that prepares and runs the reconstruction for an acquired image. This loads the image, the pre-calculated ESPIRiT coil maps, creates the undersampled version and calls the reconstruction function: vaerecon.py. After the reconstruction the reconstructed and the zero-filled images are saved as pickle files: 'rec_ddp_espirit'. 'rec_ddp_espirit' contains the images throughout the iterations, so can be used to calculate step-wise RMSE to verify convergence.
runrecon_acquired_image.py：为获取的图像准备和运行重建的文件。 这将加载图像，预先计算的ESPIRiT线圈贴图，创建欠采样版本并调用重建函数：vaerecon.py。 在重建之后，重建的和零填充的图像被保存为pickle文件：'rec_ddp_espirit'。 'rec_ddp_espirit'包含整个迭代中的图像，因此可用于计算逐步RMSE以验证收敛。

3. vaerecon.py: The main recon function. It contains the necessary functions for the recon, such as the multi-coil Fourier transforms, data projections, prior projections, phase projections etc... Also implements the POCS optimization scheme. The prior projection is the part that uses the VAE. The VAE is also called as a function here: definevae.py. This function returns the necessary operation and gradients to do the prior projection.
vaerecon.py：主要的侦察功能。 它包含了重新调整所需的功能，例如多线圈傅立叶变换，数据投影，先前投影，相位投影等......还实现了POCS优化方案。 先前的预测是使用VAE的部分。 VAE在这里也被称为函数：definevae.py。 此函数返回必要的操作和渐变以执行先前的投影。

4. definevae.py: This function contains the VAE architecture, so runs it once to generate the graph, then loads the stored values into the variables, then rewires the graph again using the stored values, but this time with a variable (x_rec) for the input image instead of a placeholder (x_inp). This is because of some issues with an earlier version of the TF while calculating derivatives w.r.t. placeholders. Finally, this implements the necessary operations and their gradients to return to the recon function.

definevae.py：此函数包含VAE体系结构，因此运行一次以生成图形，然后将存储的值加载到变量中，然后使用存储的值重新重新绘制图形，但这次使用变量（x_rec） 输入图像而不是占位符（x_inp）。 这是因为在计算衍生物w.r.t时，早期版本的TF存在一些问题。占位符。 最后，这将实现必要的操作及其渐变以返回到recon功能。

5. Patcher: A class to handle the patching operations. To avoid writing the patching functions in the recon function leading to unnecessarily cumbersome code, we made a class instead. You can make an instance of this at the very beginning, providing your image size and desired settings (patch size, overlap etc...) and use its two functions to convert an image to a set of patches and vice versa. Not very thoroughly tested. 
Patcher：一个处理修补操作的类。 为了避免在recon函数中编写修补函数导致不必要的繁琐代码，我们改为编写了一个类。 你可以在一开始就创建一个这样的实例，提供你的图像大小和所需的设置（补丁大小，重叠等...）并使用它的两个函数将图像转换为一组补丁，反之亦然。 没有经过彻底的测试。

6. US_pattern.py: A class to generate US patterns. Not thoroughly tested, but should be fine if you use it as given in the main file.
6. US_pattern.py：生成美国模式的类。 没有经过彻底的测试，但如果你按照主文件中的说明使用它应该没问题。

7. trained_model: The shared model is trained on 790 central T1 weighted slices with 1 mm in-slice resolution (158 subjects) from the HCP dataset.
7. trained_model：共享模型在来自HCP数据集的790个中心T1加权切片上训练，切片分辨率为1mm（158个对象）。
8. sample_data_and_uspat: We share a sample image from the HCP dataset* (single coil, no phase) and an image from a volunteer acquired for this study (16 coils, complex image) along with the corresponding ESPIRiT coil maps. We provide the images saved as two separate files, for the complex and imaginary parts due to complications with saving complex numbers. We also provide sample undersampling patterns for both images for R=2.
8. sample_data_and_uspat：我们共享来自HCP数据集*（单线圈，无相位）的样本图像和来自本研究获得的志愿者的图像（16个线圈，复杂图像）以及相应的ESPIRiT线圈图。 我们提供保存为两个单独文件的图像，用于复杂和虚部，因为复杂数字的复杂性。 我们还为R = 2的两个图像提供样本欠采样模式。

Note: The dependencies are given in the requirements.txt. You can use "$ conda create --name <env> --file <this file>" to create an environment with the given dependencies.



* *This image was provided by the Human Connectome Project, WU-Minn Consortium (Principal Investigators: David Van Essen and Kamil Ugurbil; 1U54MH091657) funded by the 16 NIH Institutes and Centers that support the NIH Blueprint for Neuroscience Research; and by the McDonnell Center for Systems Neuroscience at Washington University.
