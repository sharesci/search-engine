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
            let a = new metaJson();
            a.title = fileItem._file.name;
            this.uploader.options.additionalParameter = { metaJson: JSON.stringify(a)};
            fileItem.alias = "fulltextfile";
        };
    }

    uploader: FileUploader = new FileUploader({ url: URL, allowedFileType: ["pdf"] });
    hasBaseDropZoneOver: boolean = false;

    fileOverBase(e: any): void {
        this.hasBaseDropZoneOver = e;
    }
}

export class metaJson {
    constructor() { this.title = "" };
    title: string
}