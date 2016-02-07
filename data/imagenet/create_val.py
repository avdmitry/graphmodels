val_path = "./ilsvrc/val/"
val_gt_path = "./ILSVRC2014_devkit/data/ILSVRC2014_clsloc_validation_ground_truth.txt"

val_gt = open(val_gt_path, "r")
val = open("val.txt", "w")

idx = 1
for line in val_gt:
  label = int(line)-1
  val.write(val_path + "ILSVRC2012_val_" + "{:08d}".format(idx) + ".JPEG " + str(label) + "\n")
  idx += 1

val.close()
val_gt.close()
