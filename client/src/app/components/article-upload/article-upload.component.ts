import { Component, OnInit } from '@angular/core';
import { FileUploader } from 'ng2-file-upload';
import { ActivatedRoute } from '@angular/router';
const URL = '/api/v1/article';

@Component({
    templateUrl: 'src/app/components/article-upload/article-upload.component.html',
    styleUrls: ['src/app/components/article-upload/article-upload.component.css']
})

export class ArticleUploadComponent implements OnInit {
    constructor(private _route: ActivatedRoute) { }

    ngOnInit() {
        this.uploader.onBeforeUploadItem =  (fileItem) => {
            let index = fileItem.index - 1;
            this.meta[index].authors = this.metaUpdated[index].authors;
            this.meta[index].references = this.metaUpdated[index].references;
            this.meta[index].title = fileItem.file.name;
            this.uploader.options.additionalParameter = { metaJson: JSON.stringify(this.meta[index])};
            fileItem.alias = "fulltextfile";
        };

        this.uploader.onAfterAddingFile = (fileItem) => {
            if(!this.meta[fileItem.index]) {
                this.meta.push(new metaJson);
            }
        }
    }

    uploader: FileUploader = new FileUploader({ url: URL, allowedFileType: ["pdf"] });
    hasBaseDropZoneOver: boolean = false;
    meta: [metaJson] = [new metaJson()];
    metaUpdated: [metaJson] = [new metaJson()];

    fileOverBase(e: any): void {
        this.hasBaseDropZoneOver = e;
    }

    addAuthor(index: number) {
        this.meta[index].authors.push({keyname: "", forenames: ""});
        this.metaUpdated[index].authors.push({keyname: "", forenames: ""});
    }

    removeAuthor(i: number, j: number) {
        this.meta[i].authors.splice(j, 1);
        this.metaUpdated[i].authors.splice(j, 1);
    }

    updateAuthor(i: number, j: number) {
        this.metaUpdated[i].authors[j].keyname = $("#keyname" + i + "-" + j).val();
        this.metaUpdated[i].authors[j].forenames = $("#forenames" + i + "-" + j).val();
    }

    addReference(index: number) {
        this.meta[index].references.push("");
        this.metaUpdated[index].references.push("");
    }

    removeReference(i: number, j: number) {
        this.meta[i].references.splice(j, 1);
        this.metaUpdated[i].references.splice(j, 1);
    }

    updateReference(i: number, j: number) {
        this.metaUpdated[i].references[j] = $("#reference" + i + "-" + j).val();
    }

}

export class metaJson {
    constructor() { this.title = "", this.abstract = "", this.authors = [{keyname: "", forenames: ""}], this.references = [""] };
    title: string;
    abstract: string;
    authors: [{ keyname: string, forenames: string}];
    references: [string]
}