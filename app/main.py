from os import mkdir
from util import retrieve_inputs, import_image
from style_transfer import create_composition_single


def main():

    # parameterize composition model
    content_weight = 0.025
    style_weight = 5.0
    variation_weight = 1.0
    height = 50  # minimum dimensions of 48x48 pxl, VGG16 matrix requirements
    width = 50
    iterations = 1

    # retrieve all inputs
    content_names, content_paths, style_names, style_paths = retrieve_inputs()

    # for every content image
    for cn, cp in zip(content_names, content_paths):
        mkdir('./output/' + cn)
        content_np = import_image(cp, height, width)

        # create all possible content/style composition images
        for sn, sp in zip(style_names, style_paths):
            style_np = import_image(sp, height, width)
            save_name = cn + '_' + sn
            save_dir = './output/' + cn + '/' + save_name
            create_composition_single(style_np, content_np,
                                      save_name, save_dir,
                                      height, width, content_weight,
                                      style_weight, variation_weight,
                                      iterations)

if __name__ == "__main__":
    main()
