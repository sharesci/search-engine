import { Component, Inject } from '@angular/core';
import { DOCUMENT } from '@angular/platform-browser';

@Component({
    providers: [Location]
})

export class AppConfig {
    constructor(@Inject(DOCUMENT) private document: any) { 
        this.apiPort = 7080;
        this.apiUrl = this.document.location.origin;
    }

    public readonly apiUrl : string;
    public readonly apiPort : number;
};